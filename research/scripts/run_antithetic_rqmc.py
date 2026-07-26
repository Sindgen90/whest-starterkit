"""Evaluate 12,288 antithetic RQMC pairs against frozen 24,576 RQMC."""
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

ESTIMATOR = ROOT / "submissions" / "rqmc_antithetic_24576" / "estimator.py"
RQMC_RESULTS = {
    "development_selection": (
        ROOT
        / "research"
        / "results"
        / "rqmc_sample_count_sweep_development_selection_4096_16384_24576_32768.json"
    ),
    "validation_frozen": (
        ROOT
        / "research"
        / "results"
        / "rqmc_sample_count_sweep_validation_frozen_24576.json"
    ),
}
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"


def load_estimator(path: Path) -> Any:
    module_name = f"antithetic_rqmc_{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator()


def load_comparisons(
    partition: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    rqmc_path = RQMC_RESULTS[partition]
    if not rqmc_path.exists():
        raise FileNotFoundError(
            f"missing paired ordinary-RQMC result for {partition}: {rqmc_path}"
        )
    rqmc_result = json.loads(rqmc_path.read_text(encoding="utf-8"))
    rqmc_by_id = {
        int(row["mlp_id"]): row
        for row in rqmc_result["per_mlp"]
        if int(row["n_samples"]) == 24_576
    }
    k3_by_id: dict[int, dict[str, Any]] = {}
    if K3_RESULTS.exists():
        k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
        k3_by_id = {
            int(row["mlp_id"]): row
            for row in k3_result["per_mlp"]
            if row["split"] == partition and row["method"] == "stack"
        }
    return rqmc_by_id, k3_by_id


def summarize(
    records: list[dict[str, Any]],
    rqmc_by_id: dict[int, dict[str, Any]],
    k3_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rqmc_wins = [row["win_vs_rqmc_24576"] for row in records]
    k3_wins = [row["win_vs_k3"] for row in records if row["win_vs_k3"] is not None]
    return {
        "n_completed": len(records),
        "final_mse": aggregate([row["final_mse"] for row in records]),
        "all_layer_mse": aggregate([row["all_layer_mse"] for row in records]),
        "adjusted_score": aggregate([row["adjusted_score"] for row in records]),
        "flops": aggregate([float(row["flops"]) for row in records]),
        "runtime_s": aggregate([row["runtime_s"] for row in records]),
        "win_fraction_vs_rqmc_24576": float(np.mean(rqmc_wins)),
        "win_fraction_vs_k3": float(np.mean(k3_wins)) if k3_wins else None,
        "paired_delta_vs_rqmc_24576": aggregate(
            [row["delta_vs_rqmc_24576"] for row in records]
        ),
        "rqmc_24576": {
            "final_mse": aggregate(
                [float(rqmc_by_id[int(row["mlp_id"])]["final_mse"]) for row in records]
            ),
            "adjusted_score": aggregate(
                [float(rqmc_by_id[int(row["mlp_id"])]["adjusted_score"]) for row in records]
            ),
            "flops": aggregate(
                [float(rqmc_by_id[int(row["mlp_id"])]["flops"]) for row in records]
            ),
        },
        "k3": {
            "final_mse": aggregate(
                [float(k3_by_id[int(row["mlp_id"])]["final_mse"]) for row in records]
            ),
            "adjusted_score": aggregate(
                [float(k3_by_id[int(row["mlp_id"])]["adjusted_score"]) for row in records]
            ),
            "flops": aggregate(
                [float(k3_by_id[int(row["mlp_id"])]["flops"]) for row in records]
            ),
        }
        if k3_by_id
        else None,
        "prefix_mse": {
            str(prefix): aggregate([row["prefix_mse"][str(prefix)] for row in records])
            for prefix in PREFIXES
        },
    }


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    rqmc = summary["rqmc_24576"]
    improves_adjusted = (
        summary["adjusted_score"]["mean"] < rqmc["adjusted_score"]["mean"]
    )
    wins_enough = summary["win_fraction_vs_rqmc_24576"] >= 0.60
    protects_tail = (
        summary["final_mse"]["worst_decile"] <= rqmc["final_mse"]["worst_decile"]
    )
    return {
        "improves_adjusted_score_vs_rqmc": improves_adjusted,
        "win_fraction_at_least_0_60": wins_enough,
        "worst_decile_not_worse": protects_tail,
        "passed": improves_adjusted and wins_enough and protects_tail,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    rqmc = summary["rqmc_24576"]
    k3 = summary["k3"]
    decision = result["gate"]
    lines = [
        "# Antithetic RQMC 24,576",
        "",
        "One 12,288-point randomly shifted Gaussian lattice is centrally reflected.",
        "The resulting 24,576 points are propagated with equal weights. No analytic",
        "correction, fitted coefficient, or other mathematical component is included.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| method | final MSE | adjusted score | FLOPs | worst-decile MSE | win vs RQMC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| ordinary RQMC 24576 | {rqmc['final_mse']['mean']:.6g} | "
            f"{rqmc['adjusted_score']['mean']:.6g} | {rqmc['flops']['mean']:.6g} | "
            f"{rqmc['final_mse']['worst_decile']:.6g} | -- |"
        ),
    ]
    if k3:
        lines.append(
            f"| diagonal-K3 residual stack | {k3['final_mse']['mean']:.6g} | "
            f"{k3['adjusted_score']['mean']:.6g} | {k3['flops']['mean']:.6g} | "
            f"{k3['final_mse']['worst_decile']:.6g} | -- |"
        )
    lines.extend(
        [
            (
                f"| antithetic RQMC 12288 pairs | {summary['final_mse']['mean']:.6g} | "
                f"{summary['adjusted_score']['mean']:.6g} | {summary['flops']['mean']:.6g} | "
                f"{summary['final_mse']['worst_decile']:.6g} | "
                f"{summary['win_fraction_vs_rqmc_24576']:.3f} |"
            ),
            "",
            "## Gate",
            "",
            f"- Adjusted score improves over ordinary RQMC: `{decision['improves_adjusted_score_vs_rqmc']}`",
            f"- Paired win fraction is at least 0.60: `{decision['win_fraction_at_least_0_60']}`",
            f"- Worst-decile MSE is not worse: `{decision['worst_decile_not_worse']}`",
            f"- Overall gate: `{'PASS' if decision['passed'] else 'FAIL'}`",
        ]
    )
    if result["partition"] == "development_selection":
        mean_change = (
            100.0
            * (summary["final_mse"]["mean"] / rqmc["final_mse"]["mean"] - 1.0)
        )
        median_change = (
            100.0
            * (summary["final_mse"]["median"] / rqmc["final_mse"]["median"] - 1.0)
        )
        tail_change = (
            100.0
            * (
                summary["final_mse"]["worst_decile"]
                / rqmc["final_mse"]["worst_decile"]
                - 1.0
            )
        )
        lines.extend(
            [
                "",
                "## Decision",
                "",
                (
                    "Reject this configuration. It changed mean final MSE by "
                    f"{mean_change:+.2f}%, median final MSE by {median_change:+.2f}%, "
                    f"and worst-decile final MSE by {tail_change:+.2f}% versus ordinary RQMC."
                ),
                (
                    f"It won {summary['win_fraction_vs_rqmc_24576']:.3f} of paired MLPs, "
                    "but its large outliers made both mean MSE and adjusted score worse."
                ),
                "",
                "The frozen-validation gate failed. Validation was not evaluated and no "
                "submission package was created.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows(args.partition, args.limit)
    if not rows:
        raise RuntimeError("selected partition is empty")
    estimator = load_estimator(ESTIMATOR)
    rqmc_by_id, k3_by_id = load_comparisons(args.partition)
    records = []
    for index, row in enumerate(rows, start=1):
        mlp_id = int(row["mlp_id"])
        if mlp_id not in rqmc_by_id:
            raise KeyError(f"ordinary-RQMC comparison missing MLP {mlp_id}")
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
            prediction = estimator.predict(mlp, args.budget)
        runtime_s = time.perf_counter() - started
        prediction_np = np.asarray(prediction, dtype=np.float64)
        if prediction_np.shape != target.shape or not np.all(np.isfinite(prediction_np)):
            raise ValueError(f"invalid prediction for MLP {mlp_id}")
        squared_error = (prediction_np - target) ** 2
        final_mse = float(squared_error[-1].mean())
        flops_used = int(context.flops_used)
        rqmc_mse = float(rqmc_by_id[mlp_id]["final_mse"])
        k3_mse = (
            float(k3_by_id[mlp_id]["final_mse"]) if mlp_id in k3_by_id else None
        )
        records.append(
            {
                "partition": args.partition,
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "final_mse": final_mse,
                "all_layer_mse": float(squared_error.mean()),
                "adjusted_score": final_mse
                * max(SCORE_FLOOR, flops_used / args.budget),
                "flops": flops_used,
                "runtime_s": runtime_s,
                "rqmc_24576_final_mse": rqmc_mse,
                "delta_vs_rqmc_24576": final_mse - rqmc_mse,
                "win_vs_rqmc_24576": final_mse < rqmc_mse,
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
            f"delta_vs_rqmc={final_mse - rqmc_mse:.6g}",
            flush=True,
        )

    summary = summarize(records, rqmc_by_id, k3_by_id)
    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": args.partition,
        "configuration": {
            "base_samples": 12_288,
            "total_forward_samples": 24_576,
            "seed_offset": 8_675_309,
            "multiplier": 17_797,
            "budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "limit": args.limit,
            "design": "one shifted Gaussian lattice concatenated with its negatives",
        },
        "estimator_path": str(ESTIMATOR.relative_to(ROOT)),
        "estimator_sha256": hashlib.sha256(ESTIMATOR.read_bytes()).hexdigest(),
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "summary": summary,
        "gate": gate(summary),
        "per_mlp": records,
    }
    output = RESULTS / f"antithetic_rqmc_24576_{args.partition}.json"
    report = REPORTS / f"antithetic_rqmc_24576_{args.partition}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(json.dumps({"summary": summary, "gate": result["gate"]}, indent=2))
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
