"""Compare 4,096-point Box--Muller RQMC with inverse-CDF RQMC."""
from __future__ import annotations

import argparse
import hashlib
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

from run_rqmc_coset_diagnostic import BUDGET, K3_RESULTS, RQMC_SWEEP  # noqa: E402
from run_rqmc_sample_count_sweep import (  # noqa: E402
    PREFIXES,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    select_rows,
)

from submission_variants.rqmc_box_muller import estimator as candidate  # noqa: E402

RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
ESTIMATOR_PATH = (
    ROOT / "submission_variants" / "rqmc_box_muller" / "estimator.py"
)


def load_references(mlp_ids: set[int]) -> dict[str, Any]:
    sweep = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
    baseline_rows = {
        int(row["mlp_id"]): row
        for row in sweep["per_mlp"]
        if int(row["n_samples"]) == candidate.N_SAMPLES
        and int(row["mlp_id"]) in mlp_ids
    }
    k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    k3_rows = {
        int(row["mlp_id"]): row
        for row in k3_result["per_mlp"]
        if row["split"] == "development_selection"
        and row["method"] == "stack"
        and int(row["mlp_id"]) in mlp_ids
    }
    if set(baseline_rows) != mlp_ids:
        raise ValueError("inverse-CDF baseline does not cover all selected MLPs")
    if set(k3_rows) != mlp_ids:
        raise ValueError("K3 reference does not cover all selected MLPs")
    return {
        "inverse_cdf_by_id": baseline_rows,
        "k3_by_id": k3_rows,
    }


def paired_summary(
    candidate_records: list[dict[str, Any]],
    baseline_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    candidate_values = np.asarray(
        [float(row["final_mse"]) for row in candidate_records],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [
            float(baseline_by_id[int(row["mlp_id"])]["final_mse"])
            for row in candidate_records
        ],
        dtype=np.float64,
    )
    difference = candidate_values - baseline_values
    return {
        "mean_mse_difference": float(difference.mean()),
        "mse_difference": aggregate(difference.tolist()),
        "candidate_to_baseline_mse_ratio": float(
            candidate_values.mean() / baseline_values.mean()
        ),
        "gain": float(baseline_values.mean() / candidate_values.mean()),
        "win_fraction": float(np.mean(candidate_values < baseline_values)),
    }


def summarize_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_completed": len(records),
        "final_mse": aggregate([float(row["final_mse"]) for row in records]),
        "all_layer_mse": aggregate(
            [float(row["all_layer_mse"]) for row in records]
        ),
        "adjusted_score": aggregate(
            [float(row["adjusted_score"]) for row in records]
        ),
        "flops": aggregate([float(row["flops"]) for row in records]),
        "runtime_s": aggregate([float(row["runtime_s"]) for row in records]),
        "non_finite_count": int(
            sum(int(row["non_finite_count"]) for row in records)
        ),
        "prefix_mse": {
            str(prefix): aggregate(
                [float(row["prefix_mse"][str(prefix)]) for row in records]
            )
            for prefix in PREFIXES
        },
    }


def summarize_reference(
    rows_by_id: dict[int, dict[str, Any]],
) -> dict[str, float]:
    rows = list(rows_by_id.values())
    return {
        "final_mse": float(np.mean([float(row["final_mse"]) for row in rows])),
        "adjusted_score": float(
            np.mean([float(row["adjusted_score"]) for row in rows])
        ),
        "flops": float(np.mean([float(row["flops"]) for row in rows])),
        "worst_decile_mse": aggregate(
            [float(row["final_mse"]) for row in rows]
        )["worst_decile"],
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    candidate_summary = result["candidate_summary"]
    baseline = result["references"]["inverse_cdf_rqmc_4096"]
    k3 = result["references"]["k3_residual_stack"]
    paired = result["paired_vs_inverse_cdf"]
    lines = [
        "# Paired Box--Muller versus inverse-CDF RQMC at 4,096 points",
        "",
        "Only the uniform-to-Gaussian transformation changed. The rank-1 lattice,",
        "multiplier, random-shift seed policy, sample count, network forward pass,",
        "and equal-weight averaging are identical. No rotation, control variate,",
        "analytic correction, calibration, or learning is included.",
        "",
        "Only odd development-selection MLPs were evaluated. Validation and holdout",
        "were not touched.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| method | final MSE | adjusted score | FLOPs | worst-decile MSE | win vs inverse CDF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| inverse-CDF RQMC 4096 | {baseline['final_mse']:.6g} | "
            f"{baseline['adjusted_score']:.6g} | {baseline['flops']:.6g} | "
            f"{baseline['worst_decile_mse']:.6g} | -- |"
        ),
        (
            f"| Box--Muller RQMC 4096 | "
            f"{candidate_summary['final_mse']['mean']:.6g} | "
            f"{candidate_summary['adjusted_score']['mean']:.6g} | "
            f"{candidate_summary['flops']['mean']:.6g} | "
            f"{candidate_summary['final_mse']['worst_decile']:.6g} | "
            f"{paired['win_fraction']:.3f} |"
        ),
        (
            f"| K3 residual stack | {k3['final_mse']:.6g} | "
            f"{k3['adjusted_score']:.6g} | {k3['flops']:.6g} | "
            f"{k3['worst_decile_mse']:.6g} | -- |"
        ),
        "",
        "## Paired comparison",
        "",
        (
            f"- Mean paired MSE difference, Box--Muller minus inverse CDF: "
            f"`{paired['mean_mse_difference']:+.6g}`."
        ),
        (
            f"- Standard error of the paired difference: "
            f"`{paired['mse_difference']['standard_error']:.6g}`."
        ),
        f"- Raw MSE gain: `{paired['gain']:.4f}`.",
        f"- Paired win fraction: `{paired['win_fraction']:.3f}`.",
        "",
        "## Decision",
        "",
        result["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows("development_selection", args.limit)
    mlp_ids = {int(row["mlp_id"]) for row in rows}
    references_by_id = load_references(mlp_ids)
    estimator = candidate.Estimator()
    records = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            prediction = estimator.predict(mlp, BUDGET)
        runtime_s = time.perf_counter() - started
        prediction_array = np.asarray(prediction, dtype=np.float64)
        non_finite_count = int((~np.isfinite(prediction_array)).sum())
        prefix_mse = {
            str(prefix): float(
                np.mean(
                    (
                        prediction_array[min(prefix, mlp.depth) - 1]
                        - target[min(prefix, mlp.depth) - 1]
                    )
                    ** 2
                )
            )
            for prefix in PREFIXES
        }
        final_mse = float(np.mean((prediction_array[-1] - target[-1]) ** 2))
        all_layer_mse = float(np.mean((prediction_array - target) ** 2))
        score_multiplier = max(SCORE_FLOOR, context.flops_used / BUDGET)
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "final_mse": final_mse,
                "all_layer_mse": all_layer_mse,
                "adjusted_score": final_mse * score_multiplier,
                "prefix_mse": prefix_mse,
                "flops": int(context.flops_used),
                "score_multiplier": score_multiplier,
                "runtime_s": runtime_s,
                "non_finite_count": non_finite_count,
            }
        )
        print(
            f"{index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"box_muller={final_mse:.6g} "
            f"inverse={references_by_id['inverse_cdf_by_id'][int(row['mlp_id'])]['final_mse']:.6g}",
            flush=True,
        )

    candidate_summary = summarize_candidate(records)
    paired = paired_summary(
        records,
        references_by_id["inverse_cdf_by_id"],
    )
    baseline = summarize_reference(references_by_id["inverse_cdf_by_id"])
    k3 = summarize_reference(references_by_id["k3_by_id"])
    adjusted_improves = (
        candidate_summary["adjusted_score"]["mean"] < baseline["adjusted_score"]
    )
    paired_wins = paired["win_fraction"] >= 0.60
    tail_not_worse = (
        candidate_summary["final_mse"]["worst_decile"]
        <= baseline["worst_decile_mse"]
    )
    finite = candidate_summary["non_finite_count"] == 0
    gate_passed = adjusted_improves and paired_wins and tail_not_worse and finite
    relative_change = (
        candidate_summary["final_mse"]["mean"] / baseline["final_mse"] - 1.0
    )
    decision = (
        f"Box--Muller changed mean final MSE by {100.0 * relative_change:+.2f}% "
        f"versus inverse-CDF RQMC, with paired win fraction "
        f"{paired['win_fraction']:.3f}. "
    )
    if gate_passed:
        decision += (
            "It passes the development gate. Freeze this configuration and evaluate "
            "it once on validation before considering rotation or packaging."
        )
    else:
        decision += (
            "It fails the development gate. Validation remains closed; do not add an "
            "orthogonal rotation or package this estimator."
        )

    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development_selection",
        "configuration": {
            "n_samples": candidate.N_SAMPLES,
            "width_requirement": "even",
            "seed_offset": candidate.RQMC_SEED_OFFSET,
            "multiplier": candidate.RQMC_MULTIPLIER,
            "uniform_epsilon": candidate.UNIFORM_EPSILON,
            "changed_component": "inverse Gaussian CDF -> paired Box--Muller",
            "rotation": None,
            "calibration_used": False,
            "learning_used": False,
            "limit": args.limit,
            "validation_touched": False,
            "holdout_touched": False,
        },
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "source": {
            "estimator": str(ESTIMATOR_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(ESTIMATOR_PATH.read_bytes()).hexdigest(),
        },
        "candidate_summary": candidate_summary,
        "references": {
            "inverse_cdf_rqmc_4096": baseline,
            "k3_residual_stack": k3,
        },
        "paired_vs_inverse_cdf": paired,
        "gate": {
            "adjusted_score_improves": adjusted_improves,
            "paired_win_fraction_at_least_0.60": paired_wins,
            "worst_decile_not_worse": tail_not_worse,
            "no_non_finite_outputs": finite,
            "passed": gate_passed,
        },
        "decision": decision,
        "per_mlp": records,
    }
    output = RESULTS / "box_muller_rqmc_4096_development_selection.json"
    report = REPORTS / "box_muller_rqmc_4096_development_selection.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "candidate": candidate_summary,
                "references": result["references"],
                "paired": paired,
                "gate": result["gate"],
                "decision": decision,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
