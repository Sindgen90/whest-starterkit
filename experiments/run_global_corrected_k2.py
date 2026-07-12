"""Held-out global multiplicative correction for the bundled covariance K2."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import (
    BUDGET,
    DATASET,
    PREFIXES,
    RESULTS,
    REVISION,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    git_value,
    load_estimator,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "examples" / "03_covariance_propagation.py"
CALIBRATION_PARITY = 0


def raw_prediction(estimator: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mlp = as_mlp(row)
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True):
        prediction = estimator.predict(mlp, BUDGET)
    return np.asarray(prediction, dtype=np.float64), np.asarray(row["all_layer_means"], dtype=np.float64)


def evaluate(estimator: Any, row: dict[str, Any], coefficient: float) -> dict[str, Any]:
    mlp = as_mlp(row)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        raw = estimator.predict(mlp, BUDGET)
        prediction = fnp.multiply(raw, coefficient)
    runtime_s = time.perf_counter() - started
    raw = np.asarray(raw, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("invalid corrected K2 prediction")
    raw_error, corrected_error = (raw - target) ** 2, (prediction - target) ** 2
    flops_used = int(context.flops_used)
    final_mse = float(corrected_error[-1].mean())
    prefix_mse = {str(h): float(corrected_error[h - 1].mean()) for h in PREFIXES}
    return {
        "mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]), "status": "completed",
        "metrics": {"final_mse": final_mse, "all_layer_mse": float(corrected_error.mean()),
                    "prefix_mse": prefix_mse, "raw_k2_final_mse": float(raw_error[-1].mean())},
        "compute": {"flops_used": flops_used, "flop_budget": BUDGET, "runtime_s": runtime_s,
                    "flop_fraction": flops_used / BUDGET,
                    "adjusted_final_score": final_mse * max(SCORE_FLOOR, flops_used / BUDGET)},
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    return {
        "n_requested": len(records), "n_completed": len(done), "n_failed": len(records) - len(done),
        "final_mse": aggregate(r["metrics"]["final_mse"] for r in done),
        "all_layer_mse": aggregate(r["metrics"]["all_layer_mse"] for r in done),
        "raw_k2_final_mse": aggregate(r["metrics"]["raw_k2_final_mse"] for r in done),
        "adjusted_final_score": aggregate(r["compute"]["adjusted_final_score"] for r in done),
        "flops": aggregate(r["compute"]["flops_used"] for r in done),
        "runtime_s": aggregate(r["compute"]["runtime_s"] for r in done),
    }


def main() -> None:
    dataset = whestbench.load_dataset(DATASET, revision=REVISION, split="mini")
    rows = list(dataset)
    calibration = [row for row in rows if int(row["mlp_id"]) % 2 == CALIBRATION_PARITY]
    evaluation = [row for row in rows if int(row["mlp_id"]) % 2 != CALIBRATION_PARITY]
    estimator = load_estimator(BASELINE)()
    numerator = denominator = 0.0
    calibration_mses: list[float] = []
    for row in calibration:
        prediction, target = raw_prediction(estimator, row)
        numerator += float(np.vdot(prediction[-1], target[-1]))
        denominator += float(np.vdot(prediction[-1], prediction[-1]))
        calibration_mses.append(float(np.mean((prediction[-1] - target[-1]) ** 2)))
    coefficient = numerator / denominator
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in evaluation:
        try:
            records.append(evaluate(estimator, row, coefficient))
        except Exception as exc:
            records.append({"mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]), "mlp_seed": int(row["mlp_seed"]), "status": "failed"})
            failures.append({"mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]), "error_type": type(exc).__name__, "message": str(exc)})
    done = [record for record in records if record["status"] == "completed"]
    deltas = [record["metrics"]["raw_k2_final_mse"] - record["metrics"]["final_mse"] for record in done]
    result = {
        "schema_version": "1.0", "experiment": "global_multiplicative_corrected_k2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not failures else "completed_with_failures",
        "configuration": {
            "phase": "phase1", "dataset": DATASET, "dataset_revision": REVISION,
            "dataset_split": "mini", "flop_budget": BUDGET, "score_floor": SCORE_FLOOR,
            "base_estimator": str(BASELINE.relative_to(ROOT)), "depth_prefixes": list(PREFIXES),
        },
        "calibration": {
            "selector": "mlp_id % 2 == 0", "n_mlps": len(calibration),
            "objective": "final-layer MSE", "coefficient": coefficient,
            "raw_k2_final_mse": aggregate(calibration_mses),
        },
        "source": {
            "base_estimator_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
            "git_commit": git_value("rev-parse", "HEAD"),
        },
        "summary": summarize(records),
        "paired_vs_raw_k2": {
            "final_mse_reduction": aggregate(deltas),
            "win_fraction": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
        },
        "per_mlp": records, "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "global_corrected_k2_mini_holdout.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"coefficient={coefficient:.12g}; wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
