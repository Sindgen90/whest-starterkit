"""Calibrate fixed RQMC coset subsets on even development MLPs."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rqmc_coset_diagnostic import (  # noqa: E402
    BUDGET,
    N_COSETS,
    N_FULL,
    RQMC_MULTIPLIER,
    RQMC_SEED_OFFSET,
    combinations_by_size,
    combo_key,
    coset_means,
    equivalent_flops,
    selection_summary,
)
from run_rqmc_sample_count_sweep import SCORE_FLOOR, aggregate, as_mlp  # noqa: E402

from submissions.rqmc_lattice_002 import estimator as rqmc_source  # noqa: E402

MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
ODD_RESULT = (
    ROOT
    / "research"
    / "results"
    / "rqmc_24576_six_coset_diagnostic_development_selection.json"
)
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"


def select_calibration_rows(limit: int | None) -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    calibration_ids = {
        int(mlp_id)
        for mlp_id in manifest["development_mlp_ids"]
        if int(mlp_id) % 2 == 0
    }
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    selected = sorted(
        [row for row in rows if int(row["mlp_id"]) in calibration_ids],
        key=lambda row: int(row["mlp_id"]),
    )
    return selected[:limit] if limit is not None else selected


def evaluate_frozen_choices(
    selection_rows: list[dict[str, Any]],
    calibration_summary: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for size in (1, 2, 3):
        calibrated = calibration_summary[str(size)]["fixed_best"]
        key = combo_key(tuple(calibrated["combo"]))
        values = [
            float(row["combination_mse"][str(size)][key]) for row in selection_rows
        ]
        flops_used = equivalent_flops(size * 4_096)
        multiplier = max(SCORE_FLOOR, flops_used / BUDGET)
        result[str(size)] = {
            "combo": calibrated["combo"],
            "calibration_mse": calibrated["mse"],
            "selection_mse": aggregate(values),
            "selection_adjusted_score_mean": float(np.mean(values)) * multiplier,
            "equivalent_flops": flops_used,
        }
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    frozen = result["frozen_selection_results"]
    in_sample = result["odd_in_sample_best"]
    k3 = result["k3_reference"]
    lines = [
        "# Disjoint calibration check for fixed RQMC coset subsets",
        "",
        "Coset subsets were selected only on even development-calibration MLPs and",
        "then evaluated without change on the previously computed odd",
        "development-selection MLPs. Validation and holdout were not touched.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Frozen-choice matrix",
        "",
        "| size | calibrated residues | calibration MSE | selection MSE | selection adjusted | FLOPs | odd in-sample best |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for size in (1, 2, 3):
        row = frozen[str(size)]
        odd = in_sample[str(size)]
        lines.append(
            f"| {size} | `{combo_key(tuple(row['combo']))}` | "
            f"{row['calibration_mse']['mean']:.6g} | "
            f"{row['selection_mse']['mean']:.6g} | "
            f"{row['selection_adjusted_score_mean']:.6g} | "
            f"{row['equivalent_flops']:.6g} | "
            f"`{combo_key(tuple(odd['combo']))}` / {odd['mse']['mean']:.6g} |"
        )
    lines.extend(
        [
            f"| K3 | -- | -- | {k3['final_mse']:.6g} | "
            f"{k3['adjusted_score']:.6g} | {k3['flops']:.6g} | -- |",
            "",
            "## Decision",
            "",
            result["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not ODD_RESULT.exists():
        raise FileNotFoundError(ODD_RESULT)
    rows = select_calibration_rows(args.limit)
    combinations = combinations_by_size()
    rqmc_source.RQMC_SEED_OFFSET = RQMC_SEED_OFFSET
    rqmc_source.RQMC_MULTIPLIER = RQMC_MULTIPLIER
    per_mlp = []
    replay_flops = []
    runtimes = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            activations = rqmc_source.normal_lattice_samples(N_FULL, mlp.width, mlp.seed)
            for weight in mlp.weights:
                activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
        runtime_s = time.perf_counter() - started
        means = coset_means(np.asarray(activations, dtype=np.float64))
        combination_mse = {
            str(size): {
                combo_key(combo): float(
                    np.mean((means[np.asarray(combo)].mean(axis=0) - target) ** 2)
                )
                for combo in combos
            }
            for size, combos in combinations.items()
        }
        per_mlp.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "combination_mse": combination_mse,
                "replay_flops": int(context.flops_used),
                "runtime_s": runtime_s,
            }
        )
        replay_flops.append(float(context.flops_used))
        runtimes.append(runtime_s)
        print(
            f"{index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"best1={min(combination_mse['1'].values()):.6g}",
            flush=True,
        )

    calibration_summary = selection_summary(per_mlp, combinations)
    odd_result = json.loads(ODD_RESULT.read_text(encoding="utf-8"))
    odd_rows = odd_result["per_mlp"]
    frozen = evaluate_frozen_choices(odd_rows, calibration_summary)
    odd_in_sample = {
        size: odd_result["selection_summary"][size]["fixed_best"]
        for size in ("1", "2", "3")
    }
    k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    k3_rows = [
        row
        for row in k3_result["per_mlp"]
        if row["split"] == "development_selection" and row["method"] == "stack"
    ]
    k3 = {
        "final_mse": float(np.mean([row["final_mse"] for row in k3_rows])),
        "adjusted_score": float(np.mean([row["adjusted_score"] for row in k3_rows])),
        "flops": float(np.mean([row["flops"] for row in k3_rows])),
    }
    best_size = min(
        ("1", "2", "3"),
        key=lambda size: frozen[size]["selection_adjusted_score_mean"],
    )
    best = frozen[best_size]
    gap = best["selection_adjusted_score_mean"] / k3["adjusted_score"] - 1.0
    decision = (
        f"The best disjointly calibrated choice uses {best_size} cosets "
        f"`{combo_key(tuple(best['combo']))}` and has selection adjusted score "
        f"{best['selection_adjusted_score_mean']:.6g}, {100.0 * gap:+.2f}% versus K3. "
    )
    if gap < 0:
        decision += (
            "It passes this development-only calibration check and may proceed to a "
            "frozen validation evaluation before packaging."
        )
    else:
        decision += (
            "It does not beat K3, so no coset submission should be packaged from this result."
        )

    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "calibration_partition": "even development IDs",
            "selection_partition": "odd development IDs",
            "n_full": N_FULL,
            "n_cosets": N_COSETS,
            "limit": args.limit,
            "validation_touched": False,
            "holdout_touched": False,
        },
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "calibration_replay": {
            "flops": aggregate(replay_flops),
            "runtime_s": aggregate(runtimes),
        },
        "calibration_summary": calibration_summary,
        "frozen_selection_results": frozen,
        "odd_in_sample_best": odd_in_sample,
        "k3_reference": k3,
        "decision": decision,
        "per_mlp_calibration": per_mlp,
    }
    output = RESULTS / "rqmc_coset_disjoint_calibration_check.json"
    report = REPORTS / "rqmc_coset_disjoint_calibration_check.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(json.dumps({"frozen": frozen, "k3": k3, "decision": decision}, indent=2))
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
