"""Compare the nine-layer reconstruction with a phase-two 16-layer extension."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP

from puffi_reference import Estimator

BUDGET = 2**41


def evaluate(path: Path, limit: int, depths: tuple[int, ...]) -> list[dict]:
    dataset = Dataset.from_file(str(path)).with_format("numpy")
    results = []
    for index in range(min(limit, len(dataset))):
        row = dataset[index]
        weights = fnp.asarray(row["weights"], dtype=fnp.float32)
        mlp = MLP(1024, 16, [weights[i] for i in range(16)], seed=20260905)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        record = {"mlp_id": int(row["mlp_id"])}
        for depth in depths:
            estimator = Estimator()
            estimator.reference_layers = depth
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
                    prediction = np.asarray(estimator.predict(mlp, BUDGET), dtype=np.float64)
            summary = context.summary_dict()
            record[f"rich_{depth}_layer16_mse"] = float(
                np.mean((prediction[15] - target[15]) ** 2)
            )
            record[f"rich_{depth}_flops"] = int(context.flops_used)
            record[f"rich_{depth}_utilization"] = context.flops_used / BUDGET
            record[f"rich_{depth}_wall_s"] = time.perf_counter() - started
            record[f"rich_{depth}_residual_s"] = summary["residual_wall_time_s"]
        results.append(record)
        print(json.dumps(record), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--depths", type=int, nargs="+", default=(9, 16))
    args = parser.parse_args()
    with threadpool_limits(args.threads):
        depths = tuple(args.depths)
        records = evaluate(args.dataset, args.limit, depths)
    summary = {}
    for key in records[0]:
        if key != "mlp_id":
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in records]))
    for depth in depths:
        summary[f"adjusted_rich_{depth}"] = (
            summary[f"mean_rich_{depth}_layer16_mse"]
            * max(0.1, summary[f"mean_rich_{depth}_utilization"])
        )
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
