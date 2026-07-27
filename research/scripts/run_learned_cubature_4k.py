"""Train a positive learned 4k cubature rule from a fixed 24k RQMC pool."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rqmc_coset_calibration_check import select_calibration_rows  # noqa: E402
from run_rqmc_sample_count_sweep import (  # noqa: E402
    BUDGET,
    PREFIXES,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    select_rows,
)

from research.src.cubature.empirical import (  # noqa: E402
    mixed_leverage_subset,
    positive_moment_weights,
    standardize_rows,
    top_right_singular_modes,
)
from submissions.rqmc_lattice_002 import estimator as rqmc_source  # noqa: E402

N_POOL = 24_576
MAX_SUPPORT = 4_096
MODE_RANK = 256
NEURONS_PER_MLP = 16
CUBATURE_POINT_SEED = 0
SELECTION_SEED = 91_337
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
GAUSSIAN_FEATURE_WEIGHT = 4.0
RQMC_SWEEP = (
    ROOT
    / "research"
    / "results"
    / "rqmc_sample_count_sweep_development_selection_4096_16384_24576_32768.json"
)
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"


def deterministic_output_indices(width: int, mlp_seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(mlp_seed + 71_000_119)
    return np.sort(rng.choice(width, size=min(count, width), replace=False))


def selected_lattice_points(selected: np.ndarray, width: int) -> fnp.ndarray:
    """Generate only selected indices from the fixed 24k candidate lattice."""
    rng = fnp.random.default_rng(CUBATURE_POINT_SEED + RQMC_SEED_OFFSET)
    powers = []
    value = 1
    for _ in range(width):
        powers.append(float(value))
        value = (value * RQMC_MULTIPLIER) % N_POOL
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.array(selected, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    uniforms = fnp.mod(fnp.outer(indices, generator) / N_POOL + shift, 1.0)
    uniforms = fnp.clip(uniforms, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(uniforms).astype(fnp.float32)


def profile_submission_flops(mlp: Any, selected: np.ndarray, weights: fnp.ndarray) -> int:
    """Profile point construction, network propagation, and weighted reductions."""
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        activations = selected_lattice_points(selected, mlp.width)
        for weight in mlp.weights:
            activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
            _ = fnp.sum(activations * weights[:, None], axis=0)
    return int(context.flops_used)


def collect_training_snapshots(
    rows: list[dict[str, Any]],
    points: fnp.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    snapshots = np.empty((len(rows) * NEURONS_PER_MLP, N_POOL), dtype=np.float32)
    records = []
    cursor = 0
    replay_flops = []
    runtimes = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            activations = points
            for weight in mlp.weights:
                activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
        runtime_s = time.perf_counter() - started
        final = np.asarray(activations, dtype=np.float32)
        output_indices = deterministic_output_indices(
            mlp.width,
            int(row["mlp_seed"]),
            NEURONS_PER_MLP,
        )
        values = final[:, output_indices].T
        standardized, _, _ = standardize_rows(values)
        snapshots[cursor : cursor + len(output_indices)] = standardized.astype(np.float32)
        cursor += len(output_indices)
        teacher = final.mean(axis=0, dtype=np.float64)
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_seed": int(row["mlp_seed"]),
                "output_indices": output_indices.tolist(),
                "teacher_final_mse": float(np.mean((teacher - target) ** 2)),
                "replay_flops": int(context.flops_used),
                "runtime_s": runtime_s,
            }
        )
        replay_flops.append(float(context.flops_used))
        runtimes.append(runtime_s)
        print(
            f"training snapshots {index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"teacher_mse={records[-1]['teacher_final_mse']:.6g}",
            flush=True,
        )
    snapshots = snapshots[:cursor]
    return snapshots, records, {
        "flops": aggregate(replay_flops),
        "runtime_s": aggregate(runtimes),
    }


def gaussian_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.concatenate((points.T, (points * points - 1.0).T), axis=0)
    standardized, mean, scale = standardize_rows(raw)
    target = -mean / scale
    return standardized, target, {
        "pool_first_moment_max_abs": float(np.max(np.abs(points.mean(axis=0)))),
        "pool_diagonal_second_moment_max_abs": float(
            np.max(np.abs((points * points - 1.0).mean(axis=0)))
        ),
    }


def predict_selected(
    mlp: Any,
    points: fnp.ndarray,
    weights: fnp.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    learned_rows = []
    uniform_rows = []
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        activations = points
        for weight in mlp.weights:
            activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
            learned_rows.append(fnp.sum(activations * weights[:, None], axis=0))
            uniform_rows.append(fnp.mean(activations, axis=0))
    runtime_s = time.perf_counter() - started
    return (
        np.asarray(fnp.stack(learned_rows), dtype=np.float64),
        np.asarray(fnp.stack(uniform_rows), dtype=np.float64),
        int(context.flops_used),
        runtime_s,
    )


def evaluate_calibration(
    rows: list[dict[str, Any]],
    selected_points: fnp.ndarray,
    weights: fnp.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        learned, uniform, flops_used, runtime_s = predict_selected(
            mlp,
            selected_points,
            weights,
            BUDGET,
        )
        learned_error = (learned - target) ** 2
        uniform_error = (uniform - target) ** 2
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "learned_final_mse": float(learned_error[-1].mean()),
                "learned_all_layer_mse": float(learned_error.mean()),
                "uniform_final_mse": float(uniform_error[-1].mean()),
                "flops": flops_used,
                "runtime_s": runtime_s,
            }
        )
        print(
            f"calibration rule {index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"mse={records[-1]['learned_final_mse']:.6g}",
            flush=True,
        )
    return records, summarize_rule_records(records)


def evaluate_selection(
    rows: list[dict[str, Any]],
    full_points: fnp.ndarray,
    selected: np.ndarray,
    weights: fnp.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    replay_flops = []
    replay_runtimes = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        teacher_rows = []
        learned_rows = []
        uniform_rows = []
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            activations = full_points
            for weight in mlp.weights:
                activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
                selected_activations = activations[selected]
                teacher_rows.append(fnp.mean(activations, axis=0))
                learned_rows.append(
                    fnp.sum(selected_activations * weights[:, None], axis=0)
                )
                uniform_rows.append(fnp.mean(selected_activations, axis=0))
        runtime_s = time.perf_counter() - started
        teacher = np.asarray(fnp.stack(teacher_rows), dtype=np.float64)
        learned = np.asarray(fnp.stack(learned_rows), dtype=np.float64)
        uniform = np.asarray(fnp.stack(uniform_rows), dtype=np.float64)
        teacher_error = (teacher - target) ** 2
        learned_error = (learned - target) ** 2
        uniform_error = (uniform - target) ** 2
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_seed": int(row["mlp_seed"]),
                "teacher_final_mse": float(teacher_error[-1].mean()),
                "teacher_all_layer_mse": float(teacher_error.mean()),
                "learned_final_mse": float(learned_error[-1].mean()),
                "learned_all_layer_mse": float(learned_error.mean()),
                "uniform_final_mse": float(uniform_error[-1].mean()),
                "uniform_all_layer_mse": float(uniform_error.mean()),
                "replay_flops": int(context.flops_used),
                "runtime_s": runtime_s,
                "prefix_mse": {
                    str(prefix): float(learned_error[prefix - 1].mean())
                    for prefix in PREFIXES
                    if prefix <= mlp.depth
                },
            }
        )
        replay_flops.append(float(context.flops_used))
        replay_runtimes.append(runtime_s)
        print(
            f"selection rule {index}/{len(rows)} mlp_id={row['mlp_id']} "
            f"learned={records[-1]['learned_final_mse']:.6g} "
            f"teacher={records[-1]['teacher_final_mse']:.6g}",
            flush=True,
        )
    summary = summarize_rule_records(records)
    summary["teacher_final_mse"] = aggregate(
        [row["teacher_final_mse"] for row in records]
    )
    summary["uniform_all_layer_mse"] = aggregate(
        [row["uniform_all_layer_mse"] for row in records]
    )
    summary["teacher_all_layer_mse"] = aggregate(
        [row["teacher_all_layer_mse"] for row in records]
    )
    summary["replay_flops"] = aggregate(replay_flops)
    summary["replay_runtime_s"] = aggregate(replay_runtimes)
    return records, summary


def summarize_rule_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_completed": len(records),
        "learned_final_mse": aggregate(
            [row["learned_final_mse"] for row in records]
        ),
        "learned_all_layer_mse": aggregate(
            [row["learned_all_layer_mse"] for row in records]
        ),
        "uniform_final_mse": aggregate(
            [row["uniform_final_mse"] for row in records]
        ),
        "flops": aggregate([float(row.get("flops", 0.0)) for row in records]),
        "runtime_s": aggregate([row["runtime_s"] for row in records]),
    }


def load_references(selection_ids: set[int]) -> dict[str, Any]:
    rqmc_result = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
    direct_rows = [
        row
        for row in rqmc_result["per_mlp"]
        if int(row["n_samples"]) == 4_096 and int(row["mlp_id"]) in selection_ids
    ]
    teacher_rows = [
        row
        for row in rqmc_result["per_mlp"]
        if int(row["n_samples"]) == 24_576 and int(row["mlp_id"]) in selection_ids
    ]
    k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    k3_rows = [
        row
        for row in k3_result["per_mlp"]
        if row["split"] == "development_selection"
        and row["method"] == "stack"
        and int(row["mlp_id"]) in selection_ids
    ]
    return {
        "direct_rqmc_4096": {
            "final_mse": float(np.mean([row["final_mse"] for row in direct_rows])),
            "adjusted_score": float(
                np.mean([row["adjusted_score"] for row in direct_rows])
            ),
            "flops": float(np.mean([row["flops"] for row in direct_rows])),
            "by_id": {str(row["mlp_id"]): row for row in direct_rows},
        },
        "ordinary_rqmc_24576": {
            "final_mse": float(np.mean([row["final_mse"] for row in teacher_rows])),
            "adjusted_score": float(
                np.mean([row["adjusted_score"] for row in teacher_rows])
            ),
            "flops": float(np.mean([row["flops"] for row in teacher_rows])),
        },
        "k3": {
            "final_mse": float(np.mean([row["final_mse"] for row in k3_rows])),
            "adjusted_score": float(
                np.mean([row["adjusted_score"] for row in k3_rows])
            ),
            "flops": float(np.mean([row["flops"] for row in k3_rows])),
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    calibration = result["calibration_summary"]
    selection = result["selection_summary"]
    references = result["references"]
    rule = result["rule"]
    lines = [
        "# Learned positive 4k empirical cubature prototype",
        "",
        "A fixed 24,576-point Gaussian lattice pool was evaluated on even",
        "development-calibration MLPs. Sixteen deterministic final-neuron snapshots",
        "per MLP formed the training function family. A rank-256 basis, all coordinate",
        "first moments, and all diagonal second moments determined a positive weighted",
        "subset of at most 4,096 nodes. Odd development-selection MLPs were unseen",
        "during rule construction.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| method | split | final MSE | adjusted score | FLOPs |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            f"| direct RQMC 4096 | selection | "
            f"{references['direct_rqmc_4096']['final_mse']:.6g} | "
            f"{references['direct_rqmc_4096']['adjusted_score']:.6g} | "
            f"{references['direct_rqmc_4096']['flops']:.6g} |"
        ),
        (
            f"| diagonal-K3 stack | selection | {references['k3']['final_mse']:.6g} | "
            f"{references['k3']['adjusted_score']:.6g} | "
            f"{references['k3']['flops']:.6g} |"
        ),
        (
            f"| fixed 24k teacher | selection | "
            f"{selection['teacher_final_mse']['mean']:.6g} | -- | "
            f"{references['ordinary_rqmc_24576']['flops']:.6g} |"
        ),
        (
            f"| selected nodes, uniform weights | selection | "
            f"{selection['uniform_final_mse']['mean']:.6g} | "
            f"{result['scores']['uniform_adjusted']:.6g} | "
            f"{rule['submission_flops']:.6g} |"
        ),
        (
            f"| learned positive cubature | calibration | "
            f"{calibration['learned_final_mse']['mean']:.6g} | "
            f"{result['scores']['calibration_adjusted']:.6g} | "
            f"{rule['submission_flops']:.6g} |"
        ),
        (
            f"| learned positive cubature | selection | "
            f"{selection['learned_final_mse']['mean']:.6g} | "
            f"{result['scores']['selection_adjusted']:.6g} | "
            f"{rule['submission_flops']:.6g} |"
        ),
        "",
        "## Learned rule diagnostics",
        "",
        f"- Nonzero support: `{rule['support']}`.",
        f"- Effective sample size: `{rule['effective_sample_size']:.2f}`.",
        f"- Largest weight: `{rule['max_weight']:.6g}`.",
        f"- Weight solver: `{rule['solver']['mode']}`.",
        (
            "- Maximum standardized training/moment residual: "
            f"`{rule['solver']['max_constraint_residual']:.6g}`."
        ),
        f"- Maximum absolute Gaussian mean error: `{rule['gaussian_mean_max_abs']:.6g}`.",
        (
            "- Maximum absolute diagonal second-moment error: "
            f"`{rule['gaussian_diagonal_max_abs']:.6g}`."
        ),
        "",
        "## Decision",
        "",
        result["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    calibration_rows = select_calibration_rows(args.calibration_limit)
    selection_rows = select_rows("development_selection", args.selection_limit)
    widths = {
        int(np.asarray(row["weights"]).shape[-1])
        for row in calibration_rows + selection_rows
    }
    if len(widths) != 1:
        raise ValueError(f"learned rule requires one fixed input width, got {widths}")
    width = widths.pop()
    rqmc_source.RQMC_SEED_OFFSET = RQMC_SEED_OFFSET
    rqmc_source.RQMC_MULTIPLIER = RQMC_MULTIPLIER
    points_fnp = rqmc_source.normal_lattice_samples(
        N_POOL,
        width,
        CUBATURE_POINT_SEED,
    )
    points = np.asarray(points_fnp, dtype=np.float64)

    snapshots, training_records, training_replay = collect_training_snapshots(
        calibration_rows,
        points_fnp,
    )
    effective_rank = min(MODE_RANK, snapshots.shape[0])
    modes, singular_values = top_right_singular_modes(snapshots, effective_rank)
    learned_features = modes * np.sqrt(N_POOL)
    learned_target = learned_features.mean(axis=1)
    gaussian, gaussian_target, pool_moments = gaussian_features(points)
    selected_initial = mixed_leverage_subset(
        modes,
        gaussian,
        support=MAX_SUPPORT,
        seed=SELECTION_SEED,
    )
    combined_features = np.concatenate(
        (learned_features, GAUSSIAN_FEATURE_WEIGHT * gaussian),
        axis=0,
    )
    combined_target = np.concatenate(
        (learned_target, GAUSSIAN_FEATURE_WEIGHT * gaussian_target)
    )
    selected, weights_np, solver = positive_moment_weights(
        combined_features,
        combined_target,
        selected_initial,
    )
    nonzero = weights_np > 1e-12
    selected = selected[nonzero]
    weights_np = weights_np[nonzero]
    weights_np /= weights_np.sum()
    selected_points_np = points[selected].astype(np.float32)
    selected_points_fnp = fnp.array(selected_points_np)
    weights_fnp = fnp.array(weights_np.astype(np.float64))

    calibration_records, calibration_summary = evaluate_calibration(
        calibration_rows,
        selected_points_fnp,
        weights_fnp,
    )
    selection_records, selection_summary = evaluate_selection(
        selection_rows,
        points_fnp,
        selected,
        weights_fnp,
    )
    submission_flops = float(
        profile_submission_flops(
            as_mlp(calibration_rows[0]),
            selected,
            weights_fnp,
        )
    )
    selection_summary["flops"] = aggregate(
        [submission_flops for _ in selection_records]
    )
    score_multiplier = max(SCORE_FLOOR, submission_flops / BUDGET)
    scores = {
        "score_multiplier": score_multiplier,
        "calibration_adjusted": calibration_summary["learned_final_mse"]["mean"]
        * score_multiplier,
        "selection_adjusted": selection_summary["learned_final_mse"]["mean"]
        * score_multiplier,
        "uniform_adjusted": selection_summary["uniform_final_mse"]["mean"]
        * score_multiplier,
    }
    references = load_references({int(row["mlp_id"]) for row in selection_rows})
    direct_by_id = references["direct_rqmc_4096"]["by_id"]
    wins = [
        row["learned_final_mse"]
        < float(direct_by_id[str(row["mlp_id"])]["final_mse"])
        for row in selection_records
    ]
    learned_adjusted = scores["selection_adjusted"]
    k3_adjusted = references["k3"]["adjusted_score"]
    direct_adjusted = references["direct_rqmc_4096"]["adjusted_score"]
    gaussian_mean = np.sum(points[selected] * weights_np[:, None], axis=0)
    gaussian_diagonal = np.sum(
        (points[selected] * points[selected] - 1.0) * weights_np[:, None],
        axis=0,
    )
    passed = (
        learned_adjusted < min(k3_adjusted, direct_adjusted)
        and float(np.mean(wins)) >= 0.60
        and selection_summary["learned_final_mse"]["worst_decile"]
        <= selection_summary["uniform_final_mse"]["worst_decile"]
    )
    decision = (
        f"The learned rule has selection adjusted score {learned_adjusted:.6g}, "
        f"{100.0 * (learned_adjusted / k3_adjusted - 1.0):+.2f}% versus K3 and "
        f"wins {float(np.mean(wins)):.3f} of MLPs versus direct 4k RQMC. "
    )
    if passed:
        decision += (
            "It passes the development gate; freeze the rule before any validation or packaging."
        )
    else:
        decision += (
            "It fails the development gate; validation and submission packaging remain closed. "
            "Despite exact training-mode and Gaussian moment matching, the large "
            "calibration-to-selection gap identifies function-space overfitting."
        )

    rule = {
        "candidate_pool_size": N_POOL,
        "initial_support": MAX_SUPPORT,
        "support": int(len(selected)),
        "indices": selected.tolist(),
        "weights": weights_np.tolist(),
        "effective_sample_size": float(1.0 / np.sum(weights_np * weights_np)),
        "max_weight": float(weights_np.max()),
        "solver": solver,
        "mode_rank": effective_rank,
        "training_snapshot_count": int(len(snapshots)),
        "singular_values_head": singular_values[:16].tolist(),
        "submission_flops": submission_flops,
        "gaussian_mean_max_abs": float(np.max(np.abs(gaussian_mean))),
        "gaussian_diagonal_max_abs": float(np.max(np.abs(gaussian_diagonal))),
        "pool_moments": pool_moments,
    }
    rule_path = RESULTS / "learned_cubature_4k_rule.json"
    rule_path.write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "candidate_pool": "fixed 24576-point shifted Gaussian rank-1 lattice",
            "candidate_seed": CUBATURE_POINT_SEED,
            "seed_offset": RQMC_SEED_OFFSET,
            "multiplier": RQMC_MULTIPLIER,
            "max_support": MAX_SUPPORT,
            "mode_rank": effective_rank,
            "neurons_per_training_mlp": NEURONS_PER_MLP,
            "gaussian_constraints": "all coordinate first and diagonal second moments",
            "calibration_limit": args.calibration_limit,
            "selection_limit": args.selection_limit,
            "validation_touched": False,
            "holdout_touched": False,
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "training_replay": training_replay,
        "training_records": training_records,
        "rule": rule,
        "calibration_summary": calibration_summary,
        "calibration_records": calibration_records,
        "selection_summary": selection_summary,
        "selection_records": selection_records,
        "references": references,
        "scores": scores,
        "win_fraction_vs_direct_rqmc_4096": float(np.mean(wins)),
        "gate_passed": passed,
        "decision": decision,
    }
    output = RESULTS / "learned_cubature_4k_development.json"
    report = REPORTS / "learned_cubature_4k_development.md"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {rule_path.relative_to(ROOT)}")
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "rule": {key: value for key, value in rule.items() if key not in {"indices", "weights"}},
                "calibration_summary": calibration_summary,
                "selection_summary": selection_summary,
                "scores": scores,
                "win_fraction_vs_direct_rqmc_4096": float(np.mean(wins)),
                "gate_passed": passed,
                "decision": decision,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-limit", type=int, default=None)
    parser.add_argument("--selection-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_limit is not None and args.calibration_limit <= 0:
        raise SystemExit("--calibration-limit must be positive")
    if args.selection_limit is not None and args.selection_limit <= 0:
        raise SystemExit("--selection-limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
