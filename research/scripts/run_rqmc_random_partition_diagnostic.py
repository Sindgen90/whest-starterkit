"""Evaluate 50 random six-way partitions of one fixed 24,576-point lattice.

The random groups are disjoint subsets, not rank-1-lattice cosets. Candidate
selection uses even development MLPs and evaluation uses odd development MLPs.
"""
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
    K3_RESULTS,
    N_COSETS,
    N_FULL,
    N_PER_COSET,
    RQMC_MULTIPLIER,
    RQMC_SEED_OFFSET,
    RQMC_SWEEP,
    equivalent_flops,
)
from run_rqmc_sample_count_sweep import (  # noqa: E402
    SCORE_FLOOR,
    aggregate,
    as_mlp,
)

from research.src.rqmc.partition_diagnostics import (  # noqa: E402
    partition_mse,
    random_disjoint_partitions,
)
from submissions.rqmc_lattice_002 import estimator as rqmc_source  # noqa: E402

MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
TRUE_COSET_RESULT = (
    ROOT / "research" / "results" / "rqmc_coset_disjoint_calibration_check.json"
)
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
DEFAULT_PARTITIONS = 50
DEFAULT_PARTITION_SEED = 20_260_728


def select_development_rows(limit_per_split: int | None) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    development_ids = {int(value) for value in manifest["development_mlp_ids"]}
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    development = sorted(
        [row for row in rows if int(row["mlp_id"]) in development_ids],
        key=lambda row: int(row["mlp_id"]),
    )
    result = {
        "calibration": [row for row in development if int(row["mlp_id"]) % 2 == 0],
        "selection": [row for row in development if int(row["mlp_id"]) % 2 == 1],
    }
    if limit_per_split is not None:
        result = {
            split: split_rows[:limit_per_split]
            for split, split_rows in result.items()
        }
    return result


def load_references(selection_ids: set[int]) -> dict[str, Any]:
    sweep = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
    direct_rows = [
        row
        for row in sweep["per_mlp"]
        if int(row["n_samples"]) == N_PER_COSET
        and int(row["mlp_id"]) in selection_ids
    ]
    direct = {
        "final_mse": aggregate([float(row["final_mse"]) for row in direct_rows]),
        "adjusted_score_mean": float(
            np.mean([float(row["adjusted_score"]) for row in direct_rows])
        ),
        "flops": float(np.mean([float(row["flops"]) for row in direct_rows])),
    }
    k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    k3_rows = [
        row
        for row in k3_result["per_mlp"]
        if row["split"] == "development_selection"
        and row["method"] == "stack"
        and int(row["mlp_id"]) in selection_ids
    ]
    k3 = {
        "final_mse": aggregate([float(row["final_mse"]) for row in k3_rows]),
        "adjusted_score_mean": float(
            np.mean([float(row["adjusted_score"]) for row in k3_rows])
        ),
        "flops": float(np.mean([float(row["flops"]) for row in k3_rows])),
    }
    true_coset_result = json.loads(TRUE_COSET_RESULT.read_text(encoding="utf-8"))
    true_coset = true_coset_result["frozen_selection_results"]["1"]
    return {
        "direct_rqmc_4096": direct,
        "k3": k3,
        "true_residue_coset": true_coset,
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    partitions: np.ndarray,
    split: str,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    records = []
    matrices = []
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
        point_evaluation_s = time.perf_counter() - started
        reduction_started = time.perf_counter()
        mse_matrix = partition_mse(final_activations, target, partitions)
        reduction_s = time.perf_counter() - reduction_started
        full_mse = float(
            np.mean(
                (
                    final_activations.mean(axis=0, dtype=np.float64)
                    - target
                )
                ** 2
            )
        )
        matrices.append(mse_matrix)
        replay_flops.append(float(context.flops_used))
        runtimes.append(point_evaluation_s + reduction_s)
        records.append(
            {
                "split": split,
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "full_24576_mse": full_mse,
                "partition_group_mse": mse_matrix.tolist(),
                "replay_flops": int(context.flops_used),
                "point_evaluation_runtime_s": point_evaluation_s,
                "partition_reduction_runtime_s": reduction_s,
            }
        )
        print(
            f"{split} {index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"full={full_mse:.6g} random_best={mse_matrix.min():.6g}",
            flush=True,
        )
    return (
        records,
        np.stack(matrices, axis=0),
        {
            "flops": aggregate(replay_flops),
            "runtime_s": aggregate(runtimes),
        },
    )


def candidate_summary(
    calibration: np.ndarray,
    selection: np.ndarray,
) -> dict[str, Any]:
    calibration_scores = calibration.mean(axis=0)
    selection_scores = selection.mean(axis=0)
    flat_calibration = calibration_scores.reshape(-1)
    flat_selection = selection_scores.reshape(-1)
    best_flat = int(np.argmin(flat_calibration))
    best_partition, best_group = np.unravel_index(
        best_flat,
        calibration_scores.shape,
    )
    selected_per_partition = np.argmin(calibration_scores, axis=1)
    selected_partition_scores = np.asarray(
        [
            selection_scores[partition, group]
            for partition, group in enumerate(selected_per_partition)
        ],
        dtype=np.float64,
    )
    calibration_ranks = np.argsort(np.argsort(flat_calibration))
    selection_ranks = np.argsort(np.argsort(flat_selection))
    rank_correlation = float(np.corrcoef(calibration_ranks, selection_ranks)[0, 1])
    direct_flops = equivalent_flops(N_PER_COSET)
    score_multiplier = max(SCORE_FLOOR, direct_flops / BUDGET)
    selected_values = selection[:, best_partition, best_group]
    oracle_values = selection.reshape(selection.shape[0], -1).min(axis=1)
    selection_best_flat = int(np.argmin(flat_selection))
    selection_best_partition, selection_best_group = np.unravel_index(
        selection_best_flat,
        selection_scores.shape,
    )
    return {
        "n_candidates": int(flat_calibration.size),
        "equivalent_flops": direct_flops,
        "score_multiplier": score_multiplier,
        "calibration_selected": {
            "partition": int(best_partition),
            "group": int(best_group),
            "calibration_mse": float(flat_calibration[best_flat]),
            "selection_mse": aggregate(selected_values.tolist()),
            "selection_adjusted_score_mean": float(selected_values.mean())
            * score_multiplier,
            "selection_rank_of_300": int(selection_ranks[best_flat]) + 1,
        },
        "per_partition_calibration_selected_selection_mse": aggregate(
            selected_partition_scores.tolist()
        ),
        "all_candidate_selection_mse": aggregate(flat_selection.tolist()),
        "selection_in_sample_best": {
            "partition": int(selection_best_partition),
            "group": int(selection_best_group),
            "mse": float(flat_selection[selection_best_flat]),
            "adjusted_score": float(flat_selection[selection_best_flat])
            * score_multiplier,
        },
        "per_mlp_oracle_best_of_300": {
            "mse": aggregate(oracle_values.tolist()),
            "adjusted_score_mean": float(oracle_values.mean()) * score_multiplier,
        },
        "candidate_rank_spearman": rank_correlation,
        "slot_mean_calibration_mse": calibration_scores.mean(axis=0).tolist(),
        "slot_mean_selection_mse": selection_scores.mean(axis=0).tolist(),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["candidate_summary"]
    selected = summary["calibration_selected"]
    references = result["references"]
    direct = references["direct_rqmc_4096"]
    k3 = references["k3"]
    true_coset = references["true_residue_coset"]
    full = result["full_24576_selection"]
    lines = [
        "# Fifty random six-way partitions of the 24,576-point lattice",
        "",
        "Each of 50 deterministic random permutations split the same candidate pool",
        "into six disjoint 4,096-point groups, producing 300 candidate subsets.",
        "These random subsets are not rank-1-lattice cosets. The only genuine cosets",
        "of the fixed lattice are the six `k mod 6` residue classes evaluated earlier.",
        "",
        "Candidate selection used only the 30 even development MLPs. The selected",
        "subset was then evaluated unchanged on the 30 odd development MLPs.",
        "Validation and holdout were not touched.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Updated matrix",
        "",
        "| method | selection final MSE | adjusted score | equivalent FLOPs |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| direct RQMC 4096 | {direct['final_mse']['mean']:.6g} | "
            f"{direct['adjusted_score_mean']:.6g} | {direct['flops']:.6g} |"
        ),
        (
            f"| true residue coset `{'+'.join(map(str, true_coset['combo']))}` "
            f"(disjoint calibration) | {true_coset['selection_mse']['mean']:.6g} | "
            f"{true_coset['selection_adjusted_score_mean']:.6g} | "
            f"{true_coset['equivalent_flops']:.6g} |"
        ),
        (
            f"| best of 300 random subsets selected on even MLPs "
            f"`p{selected['partition']}/g{selected['group']}` | "
            f"{selected['selection_mse']['mean']:.6g} | "
            f"{selected['selection_adjusted_score_mean']:.6g} | "
            f"{summary['equivalent_flops']:.6g} |"
        ),
        (
            f"| full fixed RQMC 24576 | {full['mean']:.6g} | -- | "
            f"{equivalent_flops(N_FULL):.6g} |"
        ),
        (
            f"| K3 residual stack | {k3['final_mse']['mean']:.6g} | "
            f"{k3['adjusted_score_mean']:.6g} | {k3['flops']:.6g} |"
        ),
        "",
        "## Selection diagnostics",
        "",
        (
            f"- Calibration-selected random subset: calibration MSE "
            f"`{selected['calibration_mse']:.6g}`, selection MSE "
            f"`{selected['selection_mse']['mean']:.6g}`, selection rank "
            f"`{selected['selection_rank_of_300']}/300`."
        ),
        (
            "- Candidate calibration-versus-selection rank correlation: "
            f"`{summary['candidate_rank_spearman']:.4f}`."
        ),
        (
            "- Median selection MSE of the 50 independently calibration-selected "
            f"within-partition winners: "
            f"`{summary['per_partition_calibration_selected_selection_mse']['median']:.6g}`."
        ),
        (
            "- In-sample best random subset on odd development MLPs "
            f"`p{summary['selection_in_sample_best']['partition']}/"
            f"g{summary['selection_in_sample_best']['group']}`: "
            f"`{summary['selection_in_sample_best']['mse']:.6g}`. This is an oracle-like "
            "diagnostic, not a selectable result."
        ),
        (
            "- Per-MLP oracle choosing the best of all 300 subsets after seeing truth: "
            f"`{summary['per_mlp_oracle_best_of_300']['mse']['mean']:.6g}`."
        ),
        "",
        "## Decision",
        "",
        result["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not RQMC_SWEEP.exists():
        raise FileNotFoundError(RQMC_SWEEP)
    if not K3_RESULTS.exists():
        raise FileNotFoundError(K3_RESULTS)
    if not TRUE_COSET_RESULT.exists():
        raise FileNotFoundError(TRUE_COSET_RESULT)
    rows = select_development_rows(args.limit_per_split)
    partitions = random_disjoint_partitions(
        N_FULL,
        N_COSETS,
        args.n_partitions,
        args.partition_seed,
    )
    rqmc_source.RQMC_SEED_OFFSET = RQMC_SEED_OFFSET
    rqmc_source.RQMC_MULTIPLIER = RQMC_MULTIPLIER
    calibration_records, calibration_matrix, calibration_replay = evaluate_rows(
        rows["calibration"],
        partitions,
        "development_calibration",
    )
    selection_records, selection_matrix, selection_replay = evaluate_rows(
        rows["selection"],
        partitions,
        "development_selection",
    )
    summary = candidate_summary(calibration_matrix, selection_matrix)
    selection_ids = {int(row["mlp_id"]) for row in rows["selection"]}
    references = load_references(selection_ids)
    full_selection = aggregate(
        [float(row["full_24576_mse"]) for row in selection_records]
    )
    selected = summary["calibration_selected"]
    gap_direct = (
        selected["selection_mse"]["mean"]
        / references["direct_rqmc_4096"]["final_mse"]["mean"]
        - 1.0
    )
    gap_k3 = (
        selected["selection_adjusted_score_mean"]
        / references["k3"]["adjusted_score_mean"]
        - 1.0
    )
    decision = (
        f"The best of 300 random subsets selected on disjoint calibration MLPs has "
        f"selection MSE {selected['selection_mse']['mean']:.6g}, "
        f"{100.0 * gap_direct:+.2f}% versus direct 4096 RQMC, and adjusted score "
        f"{selected['selection_adjusted_score_mean']:.6g}, "
        f"{100.0 * gap_k3:+.2f}% versus K3. "
    )
    if gap_direct < 0 and gap_k3 < 0:
        decision += (
            "It passes the development gate, but would still need a submission-valid "
            "selected-index generator and one frozen validation run."
        )
    else:
        decision += (
            "It fails the development gate. Random partition search should stop; "
            "best-of-many oracle values are selection bias rather than a usable rule."
        )
    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "n_full": N_FULL,
            "n_groups": N_COSETS,
            "group_size": N_PER_COSET,
            "n_partitions": args.n_partitions,
            "n_candidates": args.n_partitions * N_COSETS,
            "partition_seed": args.partition_seed,
            "rqmc_seed_offset": RQMC_SEED_OFFSET,
            "rqmc_multiplier": RQMC_MULTIPLIER,
            "limit_per_split": args.limit_per_split,
            "calibration": "even development MLP IDs",
            "selection": "odd development MLP IDs",
            "random_groups_are_lattice_cosets": False,
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
        },
        "partition_indices_sha256": hashlib.sha256(partitions.tobytes()).hexdigest(),
        "calibration_replay": calibration_replay,
        "selection_replay": selection_replay,
        "candidate_summary": summary,
        "full_24576_selection": full_selection,
        "references": references,
        "decision": decision,
        "calibration_records": calibration_records,
        "selection_records": selection_records,
    }
    output = RESULTS / "rqmc_24576_fifty_random_partitions_development.json"
    report = REPORTS / "rqmc_24576_fifty_random_partitions_development.md"
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
                "candidate_summary": summary,
                "references": references,
                "full_24576_selection": full_selection,
                "decision": decision,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument("--limit-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_partitions <= 0:
        raise SystemExit("--n-partitions must be positive")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise SystemExit("--limit-per-split must be positive")
    run(args)


if __name__ == "__main__":
    main()
