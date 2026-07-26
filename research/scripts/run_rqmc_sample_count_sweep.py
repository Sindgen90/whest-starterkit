"""Sweep ordinary RQMC sample count on frozen whole-MLP partitions.

Only the number of samples changes. Each count is generated as its own complete
randomly shifted rank-1 lattice; smaller candidates are not prefixes sliced
from the largest design.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
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
RQMC = ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py"
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)
DEFAULT_COUNTS = (4096, 16384, 24576, 32768)
DEFAULT_SEED_OFFSET = 8_675_309
DEFAULT_MULTIPLIER = 17_797


def load_rqmc_estimator(n_samples: int, seed_offset: int, multiplier: int) -> Any:
    module_name = f"rqmc_count_{n_samples}_{seed_offset}_{multiplier}"
    spec = importlib.util.spec_from_file_location(module_name, RQMC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RQMC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_SAMPLES = n_samples
    module.RQMC_SEED_OFFSET = seed_offset
    module.RQMC_MULTIPLIER = multiplier
    return module.Estimator()


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
    if partition == "development_selection":
        ids = {
            int(mlp_id)
            for mlp_id in manifest["development_mlp_ids"]
            if int(mlp_id) % 2 == 1
        }
    elif partition == "validation_frozen":
        ids = {int(mlp_id) for mlp_id in manifest["validation_mlp_ids"]}
    else:
        raise ValueError(f"unsupported partition: {partition}")
    selected = sorted(
        [row for row in rows if int(row["mlp_id"]) in ids],
        key=lambda row: int(row["mlp_id"]),
    )
    return selected[:limit] if limit is not None else selected


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    worst_count = max(1, math.ceil(len(array) / 10))
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_error": float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "worst_decile": float(np.sort(array)[-worst_count:].mean()),
        "max": float(array.max()),
    }


def k3_by_mlp(partition: str) -> dict[int, dict[str, Any]]:
    if not K3_RESULTS.exists():
        return {}
    result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    return {
        int(row["mlp_id"]): row
        for row in result["per_mlp"]
        if row["split"] == partition and row["method"] == "stack"
    }


def summarize(
    records: list[dict[str, Any]],
    n_samples: int,
    baseline_by_id: dict[int, dict[str, Any]],
    k3_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    subset = [record for record in records if record["n_samples"] == n_samples]
    baseline = {int(row["mlp_id"]): row for row in records if row["n_samples"] == min(r["n_samples"] for r in records)}
    paired_baseline = [
        record["final_mse"] < baseline[int(record["mlp_id"])]["final_mse"]
        for record in subset
        if int(record["mlp_id"]) in baseline and n_samples != min(r["n_samples"] for r in records)
    ]
    paired_k3 = [
        record["final_mse"] < float(k3_rows[int(record["mlp_id"])]["final_mse"])
        for record in subset
        if int(record["mlp_id"]) in k3_rows
    ]
    paired_previous = [
        record["final_mse"] < baseline_by_id[int(record["mlp_id"])]["final_mse"]
        for record in subset
        if int(record["mlp_id"]) in baseline_by_id
    ]
    return {
        "n_samples": n_samples,
        "n_completed": len(subset),
        "final_mse": aggregate([record["final_mse"] for record in subset]),
        "all_layer_mse": aggregate([record["all_layer_mse"] for record in subset]),
        "adjusted_score": aggregate([record["adjusted_score"] for record in subset]),
        "flops": aggregate([float(record["flops"]) for record in subset]),
        "runtime_s": aggregate([record["runtime_s"] for record in subset]),
        "win_fraction_vs_4096": float(np.mean(paired_baseline)) if paired_baseline else None,
        "win_fraction_vs_k3": float(np.mean(paired_k3)) if paired_k3 else None,
        "win_fraction_vs_previous_count": float(np.mean(paired_previous)) if paired_previous else None,
        "prefix_mse": {
            str(prefix): aggregate([record["prefix_mse"][str(prefix)] for record in subset])
            for prefix in PREFIXES
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RQMC sample-count sweep",
        "",
        "Only the sample count changes. Each candidate is a complete independently constructed",
        "rank-1 lattice using the same shift rule and generator multiplier.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| partition | samples | final MSE | adjusted score | FLOPs | runtime | win vs 4096 | win vs K3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            "| {partition} | {samples} | {mse:.6g} | {score:.6g} | {flops:.6g} | "
            "{runtime:.3g}s | {win4096} | {wink3} |".format(
                partition=result["partition"],
                samples=row["n_samples"],
                mse=row["final_mse"]["mean"],
                score=row["adjusted_score"]["mean"],
                flops=row["flops"]["mean"],
                runtime=row["runtime_s"]["mean"],
                win4096="--" if row["win_fraction_vs_4096"] is None else f"{row['win_fraction_vs_4096']:.3f}",
                wink3="--" if row["win_fraction_vs_k3"] is None else f"{row['win_fraction_vs_k3']:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "No convex combination, control variate, K2 correction, or K3 correction is included.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows(args.partition, args.limit)
    counts = sorted(set(args.sample_counts))
    if not rows:
        raise RuntimeError("selected partition is empty")

    mlps = {int(row["mlp_id"]): as_mlp(row) for row in rows}
    records: list[dict[str, Any]] = []
    for count_index, n_samples in enumerate(counts, start=1):
        estimator = load_rqmc_estimator(n_samples, args.seed_offset, args.multiplier)
        for row_index, row in enumerate(rows, start=1):
            mlp_id = int(row["mlp_id"])
            mlp = mlps[mlp_id]
            target = np.asarray(row["all_layer_means"], dtype=np.float64)
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
                prediction = estimator.predict(mlp, args.budget)
            runtime_s = time.perf_counter() - started
            prediction_np = np.asarray(prediction, dtype=np.float64)
            squared_error = (prediction_np - target) ** 2
            flops_used = int(context.flops_used)
            records.append(
                {
                    "partition": args.partition,
                    "n_samples": n_samples,
                    "seed_offset": args.seed_offset,
                    "multiplier": args.multiplier,
                    "mlp_id": mlp_id,
                    "mlp_name": str(row["mlp_name"]),
                    "mlp_seed": int(row["mlp_seed"]),
                    "final_mse": float(squared_error[-1].mean()),
                    "all_layer_mse": float(squared_error.mean()),
                    "adjusted_score": float(squared_error[-1].mean())
                    * max(SCORE_FLOOR, flops_used / args.budget),
                    "flops": flops_used,
                    "runtime_s": runtime_s,
                    "prefix_mse": {
                        str(prefix): float(squared_error[prefix - 1].mean())
                        for prefix in PREFIXES
                        if prefix <= mlp.depth
                    },
                }
            )
            print(
                f"count {count_index}/{len(counts)} n={n_samples} "
                f"mlp {row_index}/{len(rows)} id={mlp_id} "
                f"mse={records[-1]['final_mse']:.6g}",
                flush=True,
            )

    baseline_by_id: dict[int, dict[str, Any]] = {}
    k3_rows = k3_by_mlp(args.partition)
    summary = []
    for n_samples in counts:
        row = summarize(records, n_samples, baseline_by_id, k3_rows)
        summary.append(row)
        baseline_by_id = {
            int(record["mlp_id"]): record
            for record in records
            if record["n_samples"] == n_samples
        }

    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": args.partition,
        "configuration": {
            "sample_counts": counts,
            "seed_offset": args.seed_offset,
            "multiplier": args.multiplier,
            "budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "limit": args.limit,
            "design_independence": "each count is constructed as its own complete shifted lattice",
        },
        "source": {
            "rqmc_path": str(RQMC.relative_to(ROOT)),
            "rqmc_sha256": hashlib.sha256(RQMC.read_bytes()).hexdigest(),
        },
        "command": " ".join(sys.argv),
        "summary": summary,
        "per_mlp": records,
    }
    tag = "_".join(str(count) for count in counts)
    output = RESULTS / f"rqmc_sample_count_sweep_{args.partition}_{tag}.json"
    report = REPORTS / f"rqmc_sample_count_sweep_{args.partition}_{tag}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        choices=("development_selection", "validation_frozen"),
        default="development_selection",
    )
    parser.add_argument("--sample-counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument("--multiplier", type=int, default=DEFAULT_MULTIPLIER)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(count <= 0 for count in args.sample_counts):
        raise SystemExit("--sample-counts must be positive")
    if args.multiplier <= 0 or args.multiplier % 2 == 0:
        raise SystemExit("--multiplier must be a positive odd integer")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
