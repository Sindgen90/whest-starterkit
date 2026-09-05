"""Sweep early aggressive HOPE merges with coupled residual correction."""
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
from run_hope_coupled_residual_scout import full_sample_final
from run_hope_merge_scout import Candidate, Merge, build_pointsets, classes, gaussian_merge_stats
from run_hope_pruning_scout import summarize_candidate
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
MAIN_SAMPLES = 32_768
STRUCTURAL_DEPTH = 20
MERGE_LAYER_COUNT = 12
ANCHORS_PER_VICTIM = 8
EPS = 1e-30

SCHEDULE_48 = (4, 4, 4, 4, 11, 9, 6, 2, 1, 1, 1, 1)
SCHEDULE_80 = (8, 8, 8, 8, 14, 12, 8, 4, 3, 3, 2, 2)
SCHEDULE_112 = (12, 12, 12, 12, 16, 14, 10, 6, 5, 5, 4, 4)


@dataclass(frozen=True)
class AggressiveCandidate:
    name: str
    n_samples: int
    correction_samples: int
    schedule: tuple[int, ...]


CANDIDATES = (
    AggressiveCandidate("early48_res4096", MAIN_SAMPLES, 4_096, SCHEDULE_48),
    AggressiveCandidate("early80_res4096", MAIN_SAMPLES, 4_096, SCHEDULE_80),
    AggressiveCandidate("early80_res8192", MAIN_SAMPLES, 8_192, SCHEDULE_80),
    AggressiveCandidate("early112_res8192", MAIN_SAMPLES, 8_192, SCHEDULE_112),
)


@dataclass(frozen=True)
class AggressivePlan:
    dead: tuple[int, ...]
    merges: tuple[Merge, ...]
    alive: tuple[int, ...]


def _pair_pool(
    layer: int,
    survivors: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    path_energy: np.ndarray,
) -> list[Merge]:
    covariance = 0.5 * (covariance + covariance.T)
    gram = covariance + np.outer(mean, mean)
    diagonal = np.maximum(np.diag(gram), EPS)
    output = []
    for victim in survivors.tolist():
        anchors = survivors[survivors != victim]
        cross = gram[victim, anchors]
        residual = np.maximum(
            diagonal[victim] - cross * cross / diagonal[anchors], 0.0
        )
        order = np.argsort(residual)[:ANCHORS_PER_VICTIM]
        for position in order:
            anchor = int(anchors[position])
            coefficient = float(cross[position] / diagonal[anchor])
            residual_mean = float(mean[victim] - coefficient * mean[anchor])
            second = float(residual[position])
            output.append(
                Merge(
                    layer=layer,
                    victim=victim,
                    anchor=anchor,
                    coefficient=coefficient,
                    residual_mean=residual_mean,
                    residual_variance=max(0.0, second - residual_mean * residual_mean),
                    residual_second_moment=second,
                    risk=second * max(float(path_energy[victim]), EPS),
                    benefit=1.0,
                )
            )
    return output


def _select(pool: list[Merge], count: int) -> tuple[Merge, ...]:
    selected = []
    used: set[int] = set()
    for merge in sorted(pool, key=lambda row: row.risk):
        if merge.victim in used or merge.anchor in used:
            continue
        selected.append(merge)
        used.add(merge.victim)
        used.add(merge.anchor)
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"selected only {len(selected)}/{count} merges")
    return tuple(selected)


def build_plans(
    mlp: Any,
    alphas: np.ndarray,
    stats: dict[str, np.ndarray],
    schedule: tuple[int, ...],
) -> tuple[AggressivePlan, ...]:
    start = mlp.depth - STRUCTURAL_DEPTH
    standard_start = mlp.depth - len(P_THRESHOLDS)
    plans = []
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        if layer < standard_start:
            dead: tuple[int, ...] = ()
        else:
            threshold_offset = layer - standard_start
            dead = tuple(
                np.flatnonzero(
                    alphas[layer] <= -P_THRESHOLDS[threshold_offset]
                ).tolist()
            )
        survivors = np.asarray(
            [index for index in range(mlp.width) if index not in set(dead)],
            dtype=np.int64,
        )
        merges: tuple[Merge, ...] = ()
        if offset < MERGE_LAYER_COUNT:
            pool = _pair_pool(
                layer,
                survivors,
                stats["means"][offset],
                stats["covariances"][offset],
                stats["path_energy"][offset],
            )
            merges = _select(pool, schedule[offset])
        removed = set(dead) | {merge.victim for merge in merges}
        alive = tuple(index for index in range(mlp.width) if index not in removed)
        plans.append(AggressivePlan(dead=dead, merges=merges, alive=alive))
    return tuple(plans)


def folded_weight(
    weight: fnp.ndarray,
    previous_plan: AggressivePlan | None,
    identity: fnp.ndarray,
) -> fnp.ndarray:
    if previous_plan is None:
        return weight
    base = weight[list(previous_plan.alive), :]
    if not previous_plan.merges:
        return base
    positions = {neuron: position for position, neuron in enumerate(previous_plan.alive)}
    victims = [merge.victim for merge in previous_plan.merges]
    anchor_positions = [positions[merge.anchor] for merge in previous_plan.merges]
    coefficients = fnp.array(
        [merge.coefficient for merge in previous_plan.merges], dtype=fnp.float32
    )
    routed = identity[anchor_positions, : len(previous_plan.alive)].T @ (
        coefficients[:, None] * weight[victims, :]
    )
    return fnp.add(base, routed)


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    result = parts[0]
    for part in parts[1:]:
        result = fnp.add(result, part)
    return result


def reduced_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    plans: tuple[AggressivePlan, ...],
    identity: fnp.ndarray,
) -> tuple[fnp.ndarray, dict[str, Any]]:
    start = mlp.depth - STRUCTURAL_DEPTH
    x = samples
    for weight in mlp.weights[:start]:
        x = fnp.maximum(x @ weight, 0.0)
    previous_plan = None
    diagnostics = {}
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        plan = plans[offset]
        weight = folded_weight(mlp.weights[layer], previous_plan, identity)
        x = fnp.maximum(x @ weight[:, plan.alive], 0.0)
        previous_plan = plan
        diagnostics[str(layer + 1)] = {
            "dead": len(plan.dead), "merges": len(plan.merges), "alive": len(plan.alive)
        }

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = classes(alphas[layer1], P_THRESHOLDS[-2])
    active2, dead2, uncertain2 = classes(alphas[layer2], P_THRESHOLDS[-1])
    w1 = folded_weight(mlp.weights[layer1], previous_plan, identity)
    w2 = mlp.weights[layer2]
    mean_x = fnp.mean(x, axis=0)
    mean1_parts = []
    h1_uncertain = None
    if active1:
        mean1_parts.append((mean_x @ w1[:, active1]) @ identity[active1, :])
    if uncertain1:
        h1_uncertain = fnp.maximum(x @ w1[:, uncertain1], 0.0)
        mean1_parts.append(fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :])
    mean1 = _sum_parts(mean1_parts, mlp.width)
    mean2_parts = []
    if active2:
        mean2_parts.append((mean1 @ w2[:, active2]) @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            z2_uncertain = x @ (w1[:, active1] @ w2[active1, :][:, uncertain2])
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
    pointset_candidates = (Candidate("base", n_samples=MAIN_SAMPLES),) + tuple(
        Candidate(f"correction_{count}", n_samples=count)
        for count in sorted({candidate.correction_samples for candidate in CANDIDATES})
    )
    pointsets = build_pointsets(pointset_candidates, width, args.sobol_direction_file)
    identity = fnp.array(np.eye(width, dtype=np.float32))
    ids, targets, k2_predictions = [], [], []
    predictions = {candidate.name: [] for candidate in CANDIDATES}
    compute = {candidate.name: [] for candidate in CANDIDATES}
    diagnostics = {candidate.name: [] for candidate in CANDIDATES}

    for row_number, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        k2, alphas, stats, k2_flops, k2_residual, stats_flops, stats_residual = (
            gaussian_merge_stats(
                mlp, dtype=fnp.float32, tail_depth=STRUCTURAL_DEPTH
            )
        )
        full_means = {}
        full_compute = {}
        for count in sorted({candidate.correction_samples for candidate in CANDIDATES}):
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                full_means[count] = full_sample_final(mlp, pointsets[count])
            full_compute[count] = (
                int(context.flops_used),
                float(context.residual_wall_time_s or 0.0),
            )

        ordered = CANDIDATES if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(CANDIDATES))
        for candidate in ordered:
            plan_started = time.perf_counter()
            plans = build_plans(mlp, alphas, stats, candidate.schedule)
            plan_time = time.perf_counter() - plan_started
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                reduced_main, candidate_diagnostics = reduced_prediction(
                    mlp, pointsets[MAIN_SAMPLES], k2, alphas, plans, identity
                )
                reduced_small, _ = reduced_prediction(
                    mlp,
                    pointsets[candidate.correction_samples],
                    k2,
                    alphas,
                    plans,
                    identity,
                )
                corrected_final = fnp.add(
                    reduced_main[-1],
                    full_means[candidate.correction_samples] - reduced_small[-1],
                )
                prediction = fnp.concatenate(
                    (k2[:-1], corrected_final[None, :]), axis=0
                )
            full_flops, full_residual = full_compute[candidate.correction_samples]
            flops_used = (
                k2_flops + stats_flops + full_flops + int(context.flops_used)
            )
            residual = (
                k2_residual
                + stats_residual
                + full_residual
                + plan_time
                + float(context.residual_wall_time_s or 0.0)
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
                "plan_time_s": plan_time,
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
                candidate,
                ids_array,
                k2_array,
                prediction_arrays[candidate.name],
                targets_array,
                compute[candidate.name],
                diagnostics[candidate.name],
                part,
                args.budget,
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
        "experiment": "hope_aggressive_residual_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET, "revision": REVISION, "split": "mini",
            "limit_mlps": args.limit_mlps, "budget": args.budget,
            "main_samples": MAIN_SAMPLES, "structural_depth": STRUCTURAL_DEPTH,
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
    stem = f"hope_aggressive_residual_limit{args.limit_mlps}"
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
