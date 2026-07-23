"""Compact development/validation grid for random-basis Hermite RQMC."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import evaluate_submission_variants as evalmod  # noqa: E402

RESULTS = ROOT / "research" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation"], default="development")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--shrinkages", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    return parser.parse_args()


def aggregate(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(np.median(arr)), se


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = whestbench.load_dataset(evalmod.DATASET, revision=evalmod.REVISION, split="mini")
    rows = evalmod.select_rows(list(dataset), args.partition, args.limit)
    if not rows:
        raise SystemExit(f"No rows selected for partition {args.partition}")

    common = SimpleNamespace(
        partition=args.partition,
        run_id=f"random-hermite-grid-{int(time.time())}",
        created_at="grid",
        budget=evalmod.BUDGET,
        score_floor=evalmod.SCORE_FLOOR,
        hermite_rank=None,
        hermite_shrinkage=None,
    )
    baseline_estimator = evalmod.load_estimator(evalmod.METHODS["rqmc_baseline"], common, "rqmc_baseline")()
    baseline = [evalmod.evaluate_one("rqmc_baseline", baseline_estimator, row, common) for row in rows]
    baseline_by_id = {record["mlp_id"]: record for record in baseline}

    output = RESULTS / f"random_hermite_grid_{args.partition}.csv"
    fieldnames = [
        "partition",
        "rank",
        "shrinkage",
        "feature_count",
        "n",
        "mean_final_mse",
        "median_final_mse",
        "se_final_mse",
        "mean_adjusted_score",
        "mean_flops",
        "mean_budget_fraction",
        "paired_mean_diff_vs_rqmc",
        "paired_median_diff_vs_rqmc",
        "win_fraction_vs_rqmc",
        "raw_mse_gain_vs_rqmc",
        "adjusted_score_gain_vs_rqmc",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank in args.ranks:
            for shrinkage in args.shrinkages:
                run_args = SimpleNamespace(**vars(common))
                run_args.hermite_rank = int(rank)
                run_args.hermite_shrinkage = float(shrinkage)
                estimator = evalmod.load_estimator(
                    evalmod.METHODS["rqmc_hermite_random"],
                    run_args,
                    "rqmc_hermite_random",
                )()
                candidate = [
                    evalmod.evaluate_one("rqmc_hermite_random", estimator, row, run_args)
                    for row in rows
                ]
                final_mean, final_median, final_se = aggregate([r["final_layer_mse"] for r in candidate])
                adjusted_mean, _, _ = aggregate([r["adjusted_challenge_score"] for r in candidate])
                flops_mean, _, _ = aggregate([r["charged_flops"] for r in candidate])
                budget_fraction_mean, _, _ = aggregate([r["budget_fraction"] for r in candidate])
                base_mse = np.asarray([baseline_by_id[r["mlp_id"]]["final_layer_mse"] for r in candidate])
                cand_mse = np.asarray([r["final_layer_mse"] for r in candidate])
                base_score = np.asarray([baseline_by_id[r["mlp_id"]]["adjusted_challenge_score"] for r in candidate])
                cand_score = np.asarray([r["adjusted_challenge_score"] for r in candidate])
                diff = cand_mse - base_mse
                writer.writerow(
                    {
                        "partition": args.partition,
                        "rank": rank,
                        "shrinkage": shrinkage,
                        "feature_count": rank + rank * (rank + 1) // 2,
                        "n": len(candidate),
                        "mean_final_mse": final_mean,
                        "median_final_mse": final_median,
                        "se_final_mse": final_se,
                        "mean_adjusted_score": adjusted_mean,
                        "mean_flops": flops_mean,
                        "mean_budget_fraction": budget_fraction_mean,
                        "paired_mean_diff_vs_rqmc": float(diff.mean()),
                        "paired_median_diff_vs_rqmc": float(np.median(diff)),
                        "win_fraction_vs_rqmc": float(np.mean(cand_mse < base_mse)),
                        "raw_mse_gain_vs_rqmc": float(base_mse.mean() / cand_mse.mean()),
                        "adjusted_score_gain_vs_rqmc": float(base_score.mean() / cand_score.mean()),
                    }
                )
                print(
                    f"rank={rank} shrink={shrinkage} mean_mse={final_mean:.6g} "
                    f"gain={base_mse.mean() / cand_mse.mean():.3f} wins={np.mean(cand_mse < base_mse):.3f}"
                )
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

