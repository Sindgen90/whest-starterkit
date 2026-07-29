"""Compare fixed partition families of one 24,576-point RQMC lattice.

This is a descriptive diagnostic. No partition or group is fitted, calibrated,
or selected using MLP performance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rqmc_coset_diagnostic import (  # noqa: E402
    BUDGET,
    N_COSETS,
    N_FULL,
    N_PER_COSET,
    RQMC_MULTIPLIER,
    RQMC_SEED_OFFSET,
    equivalent_flops,
    load_references,
)
from run_rqmc_sample_count_sweep import aggregate, as_mlp, select_rows  # noqa: E402

from research.src.rqmc.partition_diagnostics import (  # noqa: E402
    partition_means,
    random_disjoint_partitions,
    structured_six_way_partitions,
    validate_disjoint_partition,
)
from submissions.rqmc_lattice_002 import estimator as rqmc_source  # noqa: E402

RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
DEFAULT_RANDOM_PARTITIONS = 50
DEFAULT_RANDOM_SEED = 20_260_728
AFFINE_MULTIPLIER = 17_797
AFFINE_OFFSET = 8_675_309


def build_partition_collection(
    n_random: int,
    random_seed: int,
) -> tuple[list[str], np.ndarray, dict[str, list[int]]]:
    structured = structured_six_way_partitions(
        N_FULL,
        N_COSETS,
        affine_multiplier=AFFINE_MULTIPLIER,
        affine_offset=AFFINE_OFFSET,
    )
    names = list(structured)
    arrays = list(structured.values())
    family_members = {name: [index] for index, name in enumerate(names)}
    random_partitions = random_disjoint_partitions(
        N_FULL,
        N_COSETS,
        n_random,
        random_seed,
    )
    random_members = []
    for random_index, partition in enumerate(random_partitions):
        name = f"random_balanced_{random_index:02d}"
        random_members.append(len(names))
        names.append(name)
        arrays.append(partition)
    family_members["random_balanced_ensemble"] = random_members
    stacked = np.stack(arrays, axis=0)
    for partition in stacked:
        validate_disjoint_partition(partition, N_FULL)
    return names, stacked, family_members


def partition_statistics(errors: np.ndarray) -> dict[str, Any]:
    """Summarize one partition from signed group errors over MLPs and outputs."""
    if errors.ndim != 3:
        raise ValueError("errors must have shape (mlps, groups, outputs)")
    n_groups = errors.shape[1]
    flattened = errors.transpose(1, 0, 2).reshape(n_groups, -1)
    group_mse = np.mean(flattened * flattened, axis=1)
    average_individual_mse = float(group_mse.mean())
    average_error = flattened.mean(axis=0)
    full_mse = float(np.mean(average_error * average_error))
    cross = [
        float(np.mean(flattened[left] * flattened[right]))
        for left, right in itertools.combinations(range(n_groups), 2)
    ]
    average_cross_error = float(np.mean(cross))
    effective_rho = average_cross_error / average_individual_mse
    formula_mse = average_individual_mse / n_groups * (
        1.0 + (n_groups - 1) * effective_rho
    )
    leave_one_out_errors = (
        flattened.sum(axis=0)[None, :] - flattened
    ) / (n_groups - 1)
    leave_one_out_mse = np.mean(
        leave_one_out_errors * leave_one_out_errors,
        axis=1,
    )
    removal_delta = leave_one_out_mse - full_mse
    best_group = int(np.argmin(group_mse))
    largest_removal = int(np.argmax(removal_delta))
    return {
        "group_mse": group_mse.tolist(),
        "average_individual_group_mse": average_individual_mse,
        "best_group": best_group,
        "best_group_mse": float(group_mse[best_group]),
        "worst_group": int(np.argmax(group_mse)),
        "worst_group_mse": float(group_mse.max()),
        "group_mse_coefficient_of_variation": float(
            group_mse.std(ddof=1) / group_mse.mean()
        ),
        "full_six_group_mse": full_mse,
        "full_gain_vs_average_individual": 1.0
        - full_mse / average_individual_mse,
        "average_cross_error": average_cross_error,
        "effective_rho": effective_rho,
        "mse_identity_prediction": formula_mse,
        "leave_one_out_mse": leave_one_out_mse.tolist(),
        "removal_delta": removal_delta.tolist(),
        "largest_removal_penalty_group": largest_removal,
        "largest_removal_penalty": float(removal_delta[largest_removal]),
    }


def ensemble_statistics(
    partition_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "n_partitions": len(partition_summaries),
        "average_individual_group_mse": aggregate(
            [
                float(item["average_individual_group_mse"])
                for item in partition_summaries
            ]
        ),
        "best_group_mse_per_partition": aggregate(
            [float(item["best_group_mse"]) for item in partition_summaries]
        ),
        "all_group_mse": aggregate(
            [
                float(value)
                for item in partition_summaries
                for value in item["group_mse"]
            ]
        ),
        "effective_rho": aggregate(
            [float(item["effective_rho"]) for item in partition_summaries]
        ),
        "largest_removal_penalty": aggregate(
            [
                float(item["largest_removal_penalty"])
                for item in partition_summaries
            ]
        ),
        "full_six_group_mse": aggregate(
            [float(item["full_six_group_mse"]) for item in partition_summaries]
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summaries = result["partition_summaries"]
    random_summary = result["random_ensemble_summary"]
    direct = result["references"]["direct_rqmc_4096"]
    k3 = result["references"]["k3"]
    rows = [
        "# Non-learned partition-family diagnostic",
        "",
        "The same fixed 24,576-point RQMC lattice was partitioned in several",
        "predeclared ways. No MLP result was used to select a partition, group,",
        "parameter, or seed. Only odd development-selection MLPs were evaluated;",
        "validation and holdout were not touched.",
        "",
        "Every six-way partition averages back to the identical 24k estimator.",
        "The diagnostic therefore asks whether one 4k group is unusually accurate",
        "or whether the 24k gain comes from cancellation among six group errors.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| partition family | mean individual 4k MSE | best descriptive group MSE | win vs direct 4k | full six-way MSE | effective rho | largest leave-one-out penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in result["structured_names"]:
        summary = summaries[name]
        rows.append(
            f"| {name} | {summary['average_individual_group_mse']:.6g} | "
            f"{summary['best_group_mse']:.6g} (g{summary['best_group']}) | "
            f"{summary['paired_best_group_vs_direct']['win_fraction']:.3f} | "
            f"{summary['full_six_group_mse']:.6g} | "
            f"{summary['effective_rho']:.5f} | "
            f"{summary['largest_removal_penalty']:.6g} "
            f"(remove g{summary['largest_removal_penalty_group']}) |"
        )
    rows.append(
        "| 50 random balanced partitions (mean/median) | "
        f"{random_summary['average_individual_group_mse']['mean']:.6g} / "
        f"{random_summary['average_individual_group_mse']['median']:.6g} | "
        f"{random_summary['best_group_mse_per_partition']['mean']:.6g} / "
        f"{random_summary['best_group_mse_per_partition']['median']:.6g} | "
        "-- | "
        f"{random_summary['full_six_group_mse']['mean']:.6g} | "
        f"{random_summary['effective_rho']['mean']:.5f} | "
        f"{random_summary['largest_removal_penalty']['mean']:.6g} |"
    )
    rows.extend(
        [
            "",
            "Reference estimators:",
            "",
            "| estimator | final MSE | adjusted score | FLOPs |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| direct RQMC 4096 | {direct['final_mse']:.6g} | "
                f"{direct['adjusted_score']:.6g} | {direct['flops']:.6g} |"
            ),
            (
                f"| K3 residual stack | {k3['final_mse']:.6g} | "
                f"{k3['adjusted_score']:.6g} | {k3['flops']:.6g} |"
            ),
            (
                f"| fixed RQMC 24576 | {result['full_24576_mse']:.6g} | -- | "
                f"{equivalent_flops(N_FULL):.6g} |"
            ),
            "",
            "## Per-group details",
            "",
        ]
    )
    for name in result["structured_names"]:
        summary = summaries[name]
        rows.extend(
            [
                f"### {name}",
                "",
                "| group | individual MSE | MSE without group | removal delta |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for group in range(N_COSETS):
            rows.append(
                f"| {group} | {summary['group_mse'][group]:.6g} | "
                f"{summary['leave_one_out_mse'][group]:.6g} | "
                f"{summary['removal_delta'][group]:+.6g} |"
            )
        rows.append("")
    rows.extend(["## Decision", "", result["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows("development_selection", args.limit)
    names, partitions, family_members = build_partition_collection(
        args.random_partitions,
        args.random_seed,
    )
    rqmc_source.RQMC_SEED_OFFSET = RQMC_SEED_OFFSET
    rqmc_source.RQMC_MULTIPLIER = RQMC_MULTIPLIER
    signed_errors = []
    per_mlp = []
    replay_flops = []
    runtimes = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            activations = rqmc_source.normal_lattice_samples(
                N_FULL,
                mlp.width,
                mlp.seed,
            )
            for weight in mlp.weights:
                activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
        final_activations = np.asarray(activations, dtype=np.float32)
        group_means = partition_means(final_activations, partitions)
        errors = group_means - target[None, None, :]
        mse = np.mean(errors * errors, axis=2)
        leave_one_out_errors = (
            errors.sum(axis=1, keepdims=True) - errors
        ) / (N_COSETS - 1)
        leave_one_out_mse = np.mean(
            leave_one_out_errors * leave_one_out_errors,
            axis=2,
        )
        full_mse = float(np.mean(errors.mean(axis=1) ** 2))
        runtime_s = time.perf_counter() - started
        signed_errors.append(errors)
        replay_flops.append(float(context.flops_used))
        runtimes.append(runtime_s)
        per_mlp.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "full_24576_mse": full_mse,
                "group_mse_by_partition": mse.tolist(),
                "leave_one_out_mse_by_partition": leave_one_out_mse.tolist(),
                "replay_flops": int(context.flops_used),
                "runtime_s": runtime_s,
            }
        )
        print(
            f"{index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"full={full_mse:.6g}",
            flush=True,
        )

    errors_array = np.stack(signed_errors, axis=0)
    summaries = {
        name: partition_statistics(errors_array[:, partition_index])
        for partition_index, name in enumerate(names)
    }
    structured_names = [
        name for name in names if not name.startswith("random_balanced_")
    ]
    random_summaries = [
        summaries[names[index]]
        for index in family_members["random_balanced_ensemble"]
    ]
    random_summary = ensemble_statistics(random_summaries)
    direct_by_id, k3 = load_references()
    direct_rows = [direct_by_id[int(row["mlp_id"])] for row in rows]
    direct = {
        "final_mse": float(np.mean([row["final_mse"] for row in direct_rows])),
        "adjusted_score": float(
            np.mean([row["adjusted_score"] for row in direct_rows])
        ),
        "flops": float(np.mean([row["flops"] for row in direct_rows])),
    }
    references = {"direct_rqmc_4096": direct, "k3": k3}
    direct_values = np.asarray(
        [float(row["final_mse"]) for row in direct_rows],
        dtype=np.float64,
    )
    for partition_index, name in enumerate(structured_names):
        summary = summaries[name]
        group = int(summary["best_group"])
        group_values = np.mean(
            errors_array[:, partition_index, group, :] ** 2,
            axis=1,
        )
        paired_difference = group_values - direct_values
        summary["paired_best_group_vs_direct"] = {
            "mse_ratio": float(group_values.mean() / direct_values.mean()),
            "win_fraction": float(np.mean(group_values < direct_values)),
            "paired_mse_difference": aggregate(paired_difference.tolist()),
        }
    full_24576_mse = float(np.mean([row["full_24576_mse"] for row in per_mlp]))
    best_structured_name = min(
        structured_names,
        key=lambda name: summaries[name]["best_group_mse"],
    )
    best_structured = summaries[best_structured_name]
    best_paired = best_structured["paired_best_group_vs_direct"]
    best_gap = best_structured["best_group_mse"] / direct["final_mse"] - 1.0
    true_cosets = summaries["true_residue_cosets"]
    coset_averaging_gain = (
        true_cosets["average_individual_group_mse"] - full_24576_mse
    )
    max_fractional_removal = (
        true_cosets["largest_removal_penalty"] / coset_averaging_gain
    )
    decision = (
        f"The best descriptive structured 4k group is "
        f"`{best_structured_name}` group {best_structured['best_group']} with MSE "
        f"{best_structured['best_group_mse']:.6g}, {100.0 * best_gap:+.2f}% versus "
        f"direct 4096 RQMC. Its paired win fraction is "
        f"{best_paired['win_fraction']:.3f}, and its paired MSE difference is "
        f"{best_paired['paired_mse_difference']['mean']:+.6g} with standard error "
        f"{best_paired['paired_mse_difference']['standard_error']:.6g}. "
        f"The true-coset average has effective error correlation "
        f"{true_cosets['effective_rho']:.5f}; its largest leave-one-out penalty is "
        f"{true_cosets['largest_removal_penalty']:.6g} "
        f"({100.0 * max_fractional_removal:.1f}% of the reduction from the average "
        f"individual coset to the six-way mean). "
    )
    if best_gap < 0:
        decision += (
            "The small descriptive improvement is not paired-consistent and was found "
            "on this same development data. No individual group or coset is responsible "
            "for most of the 24k gain."
        )
    else:
        decision += (
            "No individual fixed group beats direct 4k. The 24k gain is produced by "
            "averaging several differently signed errors, not by one dominant coset."
        )

    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development_selection",
        "configuration": {
            "n_full": N_FULL,
            "n_groups": N_COSETS,
            "group_size": N_PER_COSET,
            "structured_families": structured_names,
            "random_partitions": args.random_partitions,
            "random_seed": args.random_seed,
            "affine_multiplier": AFFINE_MULTIPLIER,
            "affine_offset": AFFINE_OFFSET,
            "limit": args.limit,
            "performance_based_selection": False,
            "calibration_used": False,
            "validation_touched": False,
            "holdout_touched": False,
        },
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "source_sha256": {
            "rqmc_estimator": hashlib.sha256(
                (
                    ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py"
                ).read_bytes()
            ).hexdigest(),
            "partition_indices": hashlib.sha256(partitions.tobytes()).hexdigest(),
        },
        "partition_names": names,
        "structured_names": structured_names,
        "replay": {
            "flops": aggregate(replay_flops),
            "runtime_s": aggregate(runtimes),
        },
        "full_24576_mse": full_24576_mse,
        "references": references,
        "partition_summaries": summaries,
        "random_ensemble_summary": random_summary,
        "decision": decision,
        "per_mlp": per_mlp,
    }
    output = RESULTS / "rqmc_partition_family_diagnostic_development_selection.json"
    report = REPORTS / "rqmc_partition_family_diagnostic_development_selection.md"
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
                "structured": {
                    name: summaries[name] for name in structured_names
                },
                "random_ensemble": random_summary,
                "references": references,
                "decision": decision,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-partitions",
        type=int,
        default=DEFAULT_RANDOM_PARTITIONS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_partitions <= 0:
        raise SystemExit("--random-partitions must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
