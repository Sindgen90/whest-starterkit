"""Compare centered affine one- and two-anchor HOPE merge models."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, as_mlp
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_hope_merge_scout import Candidate, build_pointsets, classes, gaussian_merge_stats
from run_hope_pruning_scout import summarize_candidate
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
MERGE_COUNTS = (11, 9, 6, 2)
ANCHOR_POOL = 12
EPS = 1e-20

CANDIDATES = (
    Candidate("affine_one_anchor", selection="affine1", total_merges=sum(MERGE_COUNTS)),
    Candidate("affine_two_anchor", selection="affine2", total_merges=sum(MERGE_COUNTS)),
)


@dataclass(frozen=True)
class RegressionMerge:
    victim: int
    anchors: tuple[int, ...]
    coefficients: tuple[float, ...]
    intercept: float
    risk: float


@dataclass(frozen=True)
class RegressionPlan:
    dead: tuple[int, ...]
    merges: tuple[RegressionMerge, ...]
    alive: tuple[int, ...]


def _one_anchor_candidates(
    survivors: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    path_energy: np.ndarray,
) -> dict[int, list[RegressionMerge]]:
    variance = np.maximum(np.diag(covariance), EPS)
    output: dict[int, list[RegressionMerge]] = {}
    for victim in survivors.tolist():
        anchors = survivors[survivors != victim]
        coefficients = covariance[victim, anchors] / variance[anchors]
        residual = np.maximum(
            variance[victim] - covariance[victim, anchors] * coefficients, 0.0
        )
        order = np.argsort(residual)[:ANCHOR_POOL]
        output[victim] = [
            RegressionMerge(
                victim=victim,
                anchors=(int(anchors[position]),),
                coefficients=(float(coefficients[position]),),
                intercept=float(
                    mean[victim] - coefficients[position] * mean[anchors[position]]
                ),
                risk=float(residual[position] * max(path_energy[victim], EPS)),
            )
            for position in order
        ]
    return output


def _two_anchor_candidates(
    one_anchor: dict[int, list[RegressionMerge]],
    mean: np.ndarray,
    covariance: np.ndarray,
    path_energy: np.ndarray,
) -> list[RegressionMerge]:
    variance = np.maximum(np.diag(covariance), EPS)
    output = []
    for victim, singles in one_anchor.items():
        anchors = [row.anchors[0] for row in singles]
        candidates = []
        for left_pos, left in enumerate(anchors):
            for right in anchors[left_pos + 1 :]:
                cross = covariance[left, right]
                determinant = variance[left] * variance[right] - cross * cross
                if determinant <= EPS * variance[left] * variance[right]:
                    continue
                cov_left = covariance[victim, left]
                cov_right = covariance[victim, right]
                c_left = (cov_left * variance[right] - cov_right * cross) / determinant
                c_right = (cov_right * variance[left] - cov_left * cross) / determinant
                residual = max(
                    0.0,
                    variance[victim] - c_left * cov_left - c_right * cov_right,
                )
                candidates.append(
                    RegressionMerge(
                        victim=victim,
                        anchors=(left, right),
                        coefficients=(float(c_left), float(c_right)),
                        intercept=float(
                            mean[victim] - c_left * mean[left] - c_right * mean[right]
                        ),
                        risk=float(residual * max(path_energy[victim], EPS)),
                    )
                )
        output.extend(sorted(candidates, key=lambda row: row.risk)[:4])
    return output


def _select(candidates: list[RegressionMerge], count: int) -> tuple[RegressionMerge, ...]:
    victims: set[int] = set()
    anchors: set[int] = set()
    selected = []
    for merge in sorted(candidates, key=lambda row: row.risk):
        merge_anchors = set(merge.anchors)
        if merge.victim in victims or merge.victim in anchors or merge_anchors & victims:
            continue
        selected.append(merge)
        victims.add(merge.victim)
        anchors.update(merge_anchors)
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"selected only {len(selected)}/{count} regression merges")
    return tuple(selected)


def build_plans(
    mlp: Any,
    alphas: np.ndarray,
    stats: dict[str, np.ndarray],
    model: str,
) -> tuple[RegressionPlan, ...]:
    start = mlp.depth - len(P_THRESHOLDS)
    plans = []
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        dead = tuple(np.flatnonzero(alphas[layer] <= -P_THRESHOLDS[offset]).tolist())
        survivors = np.flatnonzero(alphas[layer] > -P_THRESHOLDS[offset])
        merges: tuple[RegressionMerge, ...] = ()
        if offset < len(MERGE_COUNTS):
            covariance = 0.5 * (
                stats["covariances"][offset] + stats["covariances"][offset].T
            )
            one_by_victim = _one_anchor_candidates(
                survivors,
                stats["means"][offset],
                covariance,
                stats["path_energy"][offset],
            )
            if model == "affine1":
                pool = [row for rows in one_by_victim.values() for row in rows]
            elif model == "affine2":
                pool = _two_anchor_candidates(
                    one_by_victim,
                    stats["means"][offset],
                    covariance,
                    stats["path_energy"][offset],
                )
            else:
                raise ValueError(model)
            merges = _select(pool, MERGE_COUNTS[offset])
        removed = set(dead) | {merge.victim for merge in merges}
        alive = tuple(index for index in range(mlp.width) if index not in removed)
        plans.append(RegressionPlan(dead=dead, merges=merges, alive=alive))
    return tuple(plans)


def folded_weight_and_bias(
    weight: fnp.ndarray,
    previous_plan: RegressionPlan | None,
    identity: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    width = weight.shape[1]
    if previous_plan is None:
        return weight, fnp.zeros(width, dtype=fnp.float32)
    base = weight[list(previous_plan.alive), :]
    if not previous_plan.merges:
        return base, fnp.zeros(width, dtype=fnp.float32)

    positions = {neuron: position for position, neuron in enumerate(previous_plan.alive)}
    anchor_positions = []
    victims = []
    coefficients = []
    intercepts = []
    intercept_victims = []
    for merge in previous_plan.merges:
        intercepts.append(merge.intercept)
        intercept_victims.append(merge.victim)
        for anchor, coefficient in zip(merge.anchors, merge.coefficients):
            anchor_positions.append(positions[anchor])
            victims.append(merge.victim)
            coefficients.append(coefficient)
    coefficient_array = fnp.array(coefficients, dtype=fnp.float32)
    routed = identity[anchor_positions, : len(previous_plan.alive)].T @ (
        coefficient_array[:, None] * weight[victims, :]
    )
    bias = fnp.array(intercepts, dtype=fnp.float32) @ weight[intercept_victims, :]
    return fnp.add(base, routed), bias


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    result = parts[0]
    for part in parts[1:]:
        result = fnp.add(result, part)
    return result


def structural_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    plans: tuple[RegressionPlan, ...],
    identity: fnp.ndarray,
) -> tuple[fnp.ndarray, dict[str, Any]]:
    start = mlp.depth - len(P_THRESHOLDS)
    x = samples
    for weight in mlp.weights[:start]:
        x = fnp.maximum(x @ weight, 0.0)

    previous_plan = None
    diagnostics = {}
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        plan = plans[offset]
        weight, bias = folded_weight_and_bias(mlp.weights[layer], previous_plan, identity)
        x = fnp.maximum(
            x @ weight[:, plan.alive] + bias[list(plan.alive)], 0.0
        )
        previous_plan = plan
        diagnostics[str(layer + 1)] = {
            "dead": len(plan.dead), "merges": len(plan.merges), "alive": len(plan.alive)
        }

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = classes(alphas[layer1], P_THRESHOLDS[-2])
    active2, dead2, uncertain2 = classes(alphas[layer2], P_THRESHOLDS[-1])
    w1, bias1 = folded_weight_and_bias(mlp.weights[layer1], previous_plan, identity)
    w2 = mlp.weights[layer2]

    mean_x = fnp.mean(x, axis=0)
    mean1_parts = []
    h1_uncertain = None
    if active1:
        mean1_parts.append(
            (mean_x @ w1[:, active1] + bias1[active1]) @ identity[active1, :]
        )
    if uncertain1:
        h1_uncertain = fnp.maximum(
            x @ w1[:, uncertain1] + bias1[uncertain1], 0.0
        )
        mean1_parts.append(fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :])
    mean1 = _sum_parts(mean1_parts, mlp.width)

    mean2_parts = []
    if active2:
        mean2_parts.append((mean1 @ w2[:, active2]) @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            active_to_uncertain = w2[active1, :][:, uncertain2]
            z2_uncertain = x @ (w1[:, active1] @ active_to_uncertain)
            z2_uncertain = fnp.add(
                z2_uncertain, bias1[active1] @ active_to_uncertain
            )
        if uncertain1:
            contribution = h1_uncertain @ w2[uncertain1, :][:, uncertain2]
            z2_uncertain = contribution if z2_uncertain is None else fnp.add(
                z2_uncertain, contribution
            )
        if z2_uncertain is not None:
            mean2_parts.append(
                fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0)
                @ identity[uncertain2, :]
            )
    mean2 = _sum_parts(mean2_parts, mlp.width)
    diagnostics[str(layer1 + 1)] = {
        "dead": len(dead1), "merges": 0, "alive": len(active1) + len(uncertain1)
    }
    diagnostics[str(layer2 + 1)] = {
        "dead": len(dead2), "merges": 0, "alive": len(active2) + len(uncertain2)
    }
    return fnp.concatenate((k2[:-1], mean2[None, :]), axis=0), diagnostics


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    width = as_mlp(rows[0]).width
    pointsets = build_pointsets(CANDIDATES, width, args.sobol_direction_file)
    samples = pointsets[CANDIDATES[0].n_samples]
    identity = fnp.array(np.eye(width, dtype=np.float32))
    ids, targets, k2_predictions = [], [], []
    predictions = {candidate.name: [] for candidate in CANDIDATES}
    compute = {candidate.name: [] for candidate in CANDIDATES}
    diagnostics = {candidate.name: [] for candidate in CANDIDATES}

    for row_number, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        k2, alphas, stats, k2_flops, k2_residual, stats_flops, stats_residual = (
            gaussian_merge_stats(mlp, dtype=fnp.float32)
        )
        plans = {}
        plan_time = {}
        for candidate in CANDIDATES:
            started = time.perf_counter()
            plans[candidate.name] = build_plans(mlp, alphas, stats, candidate.selection)
            plan_time[candidate.name] = time.perf_counter() - started

        ordered = CANDIDATES if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(CANDIDATES))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_diagnostics = structural_prediction(
                    mlp, samples, k2, alphas, plans[candidate.name], identity
                )
            flops_used = k2_flops + stats_flops + int(context.flops_used)
            residual = (
                k2_residual
                + stats_residual
                + float(context.residual_wall_time_s or 0.0)
                + plan_time[candidate.name]
            )
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            diagnostics[candidate.name].append(candidate_diagnostics)
            compute[candidate.name].append({
                "flops": float(flops_used),
                "effective_compute": float(
                    flops_used + LAMBDA_FLOPS_PER_SECOND * residual
                ),
                "residual_wall_time_s": residual,
                "runtime_s": time.perf_counter() - started,
                "plan_time_s": plan_time[candidate.name],
            })
        ids.append(int(row["mlp_id"]))
        targets.append(np.asarray(row["all_layer_means"], dtype=np.float64))
        k2_predictions.append(np.asarray(k2, dtype=np.float64))
        print(f"{row_number}/{len(rows)} {row['mlp_name']} completed", flush=True)

    ids_array = np.asarray(ids, dtype=np.int64)
    targets_array = np.stack(targets)
    k2_array = np.stack(k2_predictions)
    prediction_arrays = {name: np.stack(values) for name, values in predictions.items()}
    parity = {
        str(part): {
            candidate.name: summarize_candidate(
                candidate, ids_array, k2_array, prediction_arrays[candidate.name],
                targets_array, compute[candidate.name], diagnostics[candidate.name],
                part, args.budget,
            )
            for candidate in CANDIDATES
        }
        for part in (0, 1)
    }
    fields = (
        "combo_final_mse", "combo_all_layer_mse", "combo_adjusted_score",
        "combo_flop_only_adjusted_score", "flops", "effective_compute",
    )
    crossfit = {
        candidate.name: {
            **{
                field: 0.5 * (
                    parity["0"][candidate.name][field]["mean"]
                    + parity["1"][candidate.name][field]["mean"]
                )
                for field in fields
            },
            "lambda_mean": 0.5 * (
                parity["0"][candidate.name]["lambda"]
                + parity["1"][candidate.name]["lambda"]
            ),
        }
        for candidate in CANDIDATES
    }
    result = {
        "schema_version": "1.0",
        "experiment": "hope_regression_merge_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET, "revision": REVISION, "split": "mini",
            "limit_mlps": args.limit_mlps, "budget": args.budget,
            "merge_counts": MERGE_COUNTS, "anchor_pool": ANCHOR_POOL,
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
        },
        "summary": {"parity_crossfit": parity, "crossfit_mean": crossfit},
        "per_mlp": [
            {
                "mlp_id": int(mlp_id),
                "compute": {candidate.name: compute[candidate.name][index] for candidate in CANDIDATES},
                "diagnostics": {candidate.name: diagnostics[candidate.name][index] for candidate in CANDIDATES},
            }
            for index, mlp_id in enumerate(ids_array)
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    stem = f"hope_regression_merge_limit{args.limit_mlps}"
    json_path = RESULTS / f"{stem}.json"
    cache_path = RESULTS / f"{stem}.npz"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        cache_path, mlp_ids=ids_array, targets=targets_array, k2=k2_array,
        **{f"prediction__{name}": values for name, values in prediction_arrays.items()},
    )
    return json_path, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    json_path, cache_path = run(args)
    print(f"wrote {json_path.relative_to(ROOT)} and {cache_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
