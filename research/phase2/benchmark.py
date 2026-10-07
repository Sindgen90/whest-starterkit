"""Small, paired phase-two scout. Harness dependencies never enter a submission.

Reads one public Arrow shard, using its baked 1e9-sample targets. Uses a fixed
independent estimator RNG seed deliberately: this is not grader seed-protocol
reproduction. Evaluates every preset without fitting to the targets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

from estimator import Estimator

BUDGET = 2 ** 41
PRESETS = {
    "covariance": {"mode": "covariance"},
    "published_blend": {"mode": "blend", "samples": 4038},
    "wmc8192": {"mode": "sampling", "samples": 8192},
    "transport_l1": {"mode": "transport", "samples": 8192, "anchor_layer": 1},
    "transport_l4": {"mode": "transport", "samples": 8192, "anchor_layer": 4},
    "transport_l8": {"mode": "transport", "samples": 8192, "anchor_layer": 8},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=PRESETS, default=list(PRESETS))
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    dataset = Dataset.from_file(str(args.arrow)).with_format("numpy")
    report = {
        "dataset": "aicrowd/arc-whestbench-public-2026@v2-phase2",
        "scope": "First public Mini Arrow shard; small scout, not leaderboard replication or holdout",
        "estimator_sha256": hashlib.sha256(Path(__file__).with_name("estimator.py").read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("flopscope", "whestbench", "numpy")},
        "seed_protocol": "fixed independent estimator seed; no inference from dataset identity",
        "seed": args.seed, "budget": BUDGET, "score_floor": 0.1,
        "threads": args.threads, "presets": {m: PRESETS[m] for m in args.methods},
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(args.threads):
        for index in range(min(args.limit, len(dataset))):
            row = dataset[index]
            weights = fnp.asarray(row["weights"], dtype=fnp.float32)
            mlp = MLP(width=weights.shape[1], depth=weights.shape[0],
                      weights=[weights[j] for j in range(weights.shape[0])], seed=args.seed)
            truth = np.asarray(row["all_layer_means"], dtype=np.float64)
            for name in args.methods:
                estimator = Estimator()
                for key, value in PRESETS[name].items():
                    setattr(estimator, key, value)
                record = {"mlp_id": int(row["mlp_id"]), "method": name}
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
                            prediction = estimator.predict(mlp, BUDGET)
                    wall = time.perf_counter() - start
                    prediction = np.asarray(prediction, dtype=np.float64)
                    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                        raise ValueError("Invalid prediction")
                    timing = context.summary_dict()
                    residual = timing["residual_wall_time_s"]
                    mse = float(np.mean((prediction[-1] - truth[-1]) ** 2))
                    compute = context.flops_used
                    valid = compute <= BUDGET and wall <= 120 and residual <= 0.4
                    record.update(raw_mse=mse, all_layer_mse=float(np.mean((prediction-truth)**2)),
                                  flops=compute, utilization=compute/BUDGET, wall_s=wall,
                                  residual_s=residual, local_caps_pass=valid,
                                  nominal_adjusted=mse * max(0.1, compute/BUDGET),
                                  cap_aware_score=mse * max(0.1, compute/BUDGET) if valid
                                  else float(np.mean(truth[-1] ** 2)))
                except Exception as exc:
                    record.update(error=repr(exc), local_caps_pass=False,
                                  cap_aware_score=float(np.mean(truth[-1] ** 2)))
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(record), flush=True)
    report["summary"] = {}
    for name in args.methods:
        records = [r for r in report["records"] if r["method"] == name]
        good = [r for r in records if "raw_mse" in r]
        report["summary"][name] = {
            "count": len(records), "failures": sum(not r["local_caps_pass"] for r in records),
            "mean_cap_aware_score": float(np.mean([r["cap_aware_score"] for r in records])),
            **{f"mean_{field}": float(np.mean([r[field] for r in good])) if good else None
               for field in ("raw_mse", "nominal_adjusted", "utilization", "wall_s", "residual_s")},
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
