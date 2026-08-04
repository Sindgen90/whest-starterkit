"""Compare v0.10 operation ledgers for the packaged and scout HOPE paths."""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from run_baselines import DATASET, REVISION, as_mlp
from run_hope_merge_optimized_scout import (
    OPTIMIZED,
    gaussian_top_pair_payload,
    optimized_plans,
)
from run_hope_merge_scout import structural_prediction

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submissions" / "structural_tail16_hope_merge_static_float32_sobol32768"


def load_estimator_module():
    spec = importlib.util.spec_from_file_location("profiled_estimator", SUBMISSION / "estimator.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load estimator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(context: flops.BudgetContext) -> dict[str, int]:
    grouped: dict[str, int] = defaultdict(int)
    for record in context.op_log:
        grouped[record.op_name] += int(record.flop_cost)
    return dict(sorted(grouped.items(), key=lambda row: row[1], reverse=True))


def main() -> None:
    row = next(iter(whestbench.load_dataset(DATASET, revision=REVISION, split="mini")))
    mlp = as_mlp(row)
    module = load_estimator_module()
    payload = fnp.load(str(SUBMISSION / module.SAMPLES_FILE))
    samples = payload["samples"]
    identity = payload["identity"]

    with flops.BudgetContext(flop_budget=10**15, quiet=True) as package_k2:
        k2, tail_alphas, pair_payload = module._gaussian_classifier(mlp)
    plans = module._merge_plans(mlp, tail_alphas, pair_payload)
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as package_structural:
        structural = module._structural_prediction(
            mlp, samples, k2, tail_alphas, plans, identity
        )
        fnp.add(
            fnp.multiply(k2, 1.0 - module.LAMBDA),
            fnp.multiply(structural, module.LAMBDA),
        )

    (
        scout_k2,
        scout_alphas,
        scout_payload,
        scout_k2_flops,
        _,
        scout_merge_flops,
        _,
    ) = gaussian_top_pair_payload(mlp, dtype=fnp.float32)
    scout_plans = optimized_plans(mlp, scout_alphas, scout_payload)
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as scout_structural:
        structural_prediction(
            mlp,
            samples,
            scout_k2,
            scout_alphas,
            {},
            scout_plans,
            identity,
            OPTIMIZED,
        )

    print("package_k2", package_k2.flops_used)
    print("package_structural", package_structural.flops_used)
    print("package_total", package_k2.flops_used + package_structural.flops_used)
    print("package_k2_ops", summarize(package_k2))
    print("package_structural_ops", summarize(package_structural))
    print("scout_k2", scout_k2_flops)
    print("scout_merge", scout_merge_flops)
    print("scout_structural", scout_structural.flops_used)
    print("scout_total", scout_k2_flops + scout_merge_flops + scout_structural.flops_used)
    print("scout_structural_ops", summarize(scout_structural))
    print("prediction_delta", float(fnp.max(fnp.abs(k2 - scout_k2))))


if __name__ == "__main__":
    main()
