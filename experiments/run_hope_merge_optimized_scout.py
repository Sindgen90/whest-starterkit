"""Test a submission-shaped remote top-pair HOPE merge planner."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, as_mlp
from run_deep_structural_tail_scout import K2_COEFFICIENT
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_hope_merge_scout import (
    Candidate,
    LayerPlan,
    Merge,
    build_pointsets,
    structural_prediction,
)
from run_hope_pruning_scout import summarize_candidate
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 32_768
STATIC_MERGE_COUNTS = (11, 9, 6, 2)
ANCHORS_PER_VICTIM = 4
TOP_PAIRS_PER_LAYER = 256
MERGE_LAYER_COUNT = len(STATIC_MERGE_COUNTS)
_COV_RESCALE_THRESHOLD = 1e100

CONTROL = Candidate("control_dead_m050")
OPTIMIZED = Candidate(
    "merge_static_11_09_06_02_top256",
    n_samples=N_SAMPLES,
    selection="static",
    total_merges=sum(STATIC_MERGE_COUNTS),
)
CANDIDATES = (CONTROL, OPTIMIZED)


def gaussian_top_pair_payload(
    mlp: Any,
    dtype: Any = fnp.float64,
) -> tuple[fnp.ndarray, np.ndarray, np.ndarray, int, float, int, float]:
    """Compute K2 and materialize only top merge-pair records for layers 17-20."""
    start = mlp.depth - len(P_THRESHOLDS)
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
        mu = fnp.zeros(mlp.width, dtype=dtype)
        cov = fnp.eye(mlp.width, dtype=dtype)
        log_scale = 0.0
        rows: list[fnp.ndarray] = []
        alphas: list[fnp.ndarray] = []
        gains: list[fnp.ndarray] = []
        merge_means: list[fnp.ndarray] = []
        merge_covariances: list[fnp.ndarray] = []
        merge_log_scales: list[float] = []

        for layer, weight in enumerate(mlp.weights):
            max_var = float(fnp.max(fnp.diag(cov)))
            if max_var > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var))
                mu = mu / scale
                cov = cov / (scale * scale)
                log_scale += float(fnp.log(scale))

            mu_pre = weight.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, weight, weight)
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma
            phi = flops.stats.norm.pdf(alpha)
            gain = flops.stats.norm.cdf(alpha)
            mu = mu_pre * gain + sigma * phi
            second = (mu_pre * mu_pre + var_pre) * gain + mu_pre * sigma * phi
            var_post = fnp.maximum(second - mu * mu, 0.0)
            gain_work = fnp.asarray(gain, dtype=dtype)
            cov = fnp.multiply(fnp.outer(gain_work, gain_work), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            layer_scale = float(fnp.exp(log_scale))
            raw_mean = mu * layer_scale
            rows.append(raw_mean)
            alphas.append(alpha)
            gains.append(gain_work)
            if start <= layer < start + MERGE_LAYER_COUNT:
                merge_means.append(raw_mean)
                merge_covariances.append(cov)
                merge_log_scales.append(log_scale)

        k2 = fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT)
        alpha_values = np.asarray(fnp.stack(alphas, axis=0), dtype=np.float64)

    with flops.BudgetContext(flop_budget=10**15, quiet=True) as merge_context:
        path_energy: list[fnp.ndarray | None] = [None] * mlp.depth
        path = fnp.ones(mlp.width) / mlp.width
        path_energy[-1] = path
        for layer in range(mlp.depth - 2, -1, -1):
            path = (mlp.weights[layer + 1] ** 2) @ (
                gains[layer + 1] ** 2 * path
            )
            path_energy[layer] = path

        payloads = []
        for offset, count in enumerate(STATIC_MERGE_COUNTS):
            layer = start + offset
            layer_scale2 = math.exp(2.0 * merge_log_scales[offset])
            covariance = merge_covariances[offset] * layer_scale2
            covariance = 0.5 * (covariance + covariance.T)
            mean = merge_means[offset]
            gram = covariance + fnp.outer(mean, mean)
            diagonal = fnp.maximum(fnp.diag(gram), 1e-30)
            coefficient = gram / diagonal[None, :]
            residual_mean = mean[:, None] - coefficient * mean[None, :]
            residual_second = fnp.maximum(
                diagonal[:, None] - gram * gram / diagonal[None, :], 0.0
            )
            residual_variance = fnp.maximum(
                residual_second - residual_mean * residual_mean, 0.0
            )
            risk = residual_second * path_energy[layer][:, None]

            alive = alphas[layer] > -P_THRESHOLDS[offset]
            valid = alive[:, None] & alive[None, :]
            valid = valid & (~fnp.eye(mlp.width, dtype=fnp.bool_))
            risk = fnp.where(valid, risk, float("inf"))
            anchor_order = fnp.argsort(risk, axis=1)[:, :ANCHORS_PER_VICTIM]
            candidate_risk = fnp.take_along_axis(risk, anchor_order, axis=1)
            candidate_coefficient = fnp.take_along_axis(
                coefficient, anchor_order, axis=1
            )
            candidate_mean = fnp.take_along_axis(
                residual_mean, anchor_order, axis=1
            )
            candidate_variance = fnp.take_along_axis(
                residual_variance, anchor_order, axis=1
            )
            candidate_second = fnp.take_along_axis(
                residual_second, anchor_order, axis=1
            )
            victim_grid = fnp.broadcast_to(
                fnp.arange(mlp.width)[:, None], anchor_order.shape
            )
            order = fnp.argsort(fnp.ravel(candidate_risk))[
                :TOP_PAIRS_PER_LAYER
            ]
            victims = fnp.take(fnp.ravel(victim_grid), order)
            anchors = fnp.take(fnp.ravel(anchor_order), order)
            layer_values = fnp.full(
                TOP_PAIRS_PER_LAYER, layer, dtype=fnp.float64
            )
            payloads.append(
                fnp.stack(
                    (
                        layer_values,
                        victims.astype(fnp.float64),
                        anchors.astype(fnp.float64),
                        fnp.take(fnp.ravel(candidate_coefficient), order),
                        fnp.take(fnp.ravel(candidate_mean), order),
                        fnp.take(fnp.ravel(candidate_variance), order),
                        fnp.take(fnp.ravel(candidate_second), order),
                        fnp.take(fnp.ravel(candidate_risk), order),
                    ),
                    axis=1,
                )
            )
            if count * 2 > mlp.width:
                raise RuntimeError("static merge count exceeds disjoint capacity")
        payload = np.asarray(fnp.concatenate(payloads, axis=0), dtype=np.float64)

    return (
        k2,
        alpha_values,
        payload,
        int(k2_context.flops_used),
        float(k2_context.residual_wall_time_s or 0.0),
        int(merge_context.flops_used),
        float(merge_context.residual_wall_time_s or 0.0),
    )


def control_plans(mlp: Any, alphas: np.ndarray) -> tuple[LayerPlan, ...]:
    start = mlp.depth - len(P_THRESHOLDS)
    plans = []
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        dead = tuple(np.flatnonzero(alphas[layer] <= -P_THRESHOLDS[offset]).tolist())
        dead_set = set(dead)
        alive = tuple(index for index in range(mlp.width) if index not in dead_set)
        plans.append(LayerPlan(dead, (), (), alive))
    return tuple(plans)


def optimized_plans(
    mlp: Any, alphas: np.ndarray, payload: np.ndarray
) -> tuple[LayerPlan, ...]:
    start = mlp.depth - len(P_THRESHOLDS)
    selected_by_layer: dict[int, tuple[Merge, ...]] = {}
    for offset, count in enumerate(STATIC_MERGE_COUNTS):
        layer = start + offset
        rows = payload[payload[:, 0] == layer]
        used: set[int] = set()
        selected = []
        for row in rows:
            victim = int(row[1])
            anchor = int(row[2])
            if victim in used or anchor in used:
                continue
            selected.append(
                Merge(
                    layer=layer,
                    victim=victim,
                    anchor=anchor,
                    coefficient=float(row[3]),
                    residual_mean=float(row[4]),
                    residual_variance=float(row[5]),
                    residual_second_moment=float(row[6]),
                    risk=float(row[7]),
                    benefit=1.0,
                )
            )
            used.add(victim)
            used.add(anchor)
            if len(selected) >= count:
                break
        if len(selected) != count:
            raise RuntimeError(
                f"top-{TOP_PAIRS_PER_LAYER} payload yielded only "
                f"{len(selected)}/{count} disjoint merges at layer {layer + 1}"
            )
        selected_by_layer[layer] = tuple(selected)

    plans = []
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        dead = tuple(np.flatnonzero(alphas[layer] <= -P_THRESHOLDS[offset]).tolist())
        merges = selected_by_layer.get(layer, ())
        removed = set(dead) | {merge.victim for merge in merges}
        alive = tuple(index for index in range(mlp.width) if index not in removed)
        plans.append(LayerPlan(dead, merges, (), alive))
    return tuple(plans)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    width = as_mlp(rows[0]).width
    pointsets = build_pointsets(CANDIDATES, width, args.sobol_direction_file)
    identity = fnp.array(np.eye(width, dtype=np.float32))

    ids = []
    targets = []
    k2_predictions = []
    predictions = {candidate.name: [] for candidate in CANDIDATES}
    compute = {candidate.name: [] for candidate in CANDIDATES}
    diagnostics = {candidate.name: [] for candidate in CANDIDATES}
    k2_dtype = fnp.float32 if args.k2_dtype == "float32" else fnp.float64

    for row_number, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        (
            k2,
            alphas,
            payload,
            k2_flops,
            k2_residual,
            merge_flops,
            merge_residual,
        ) = gaussian_top_pair_payload(mlp, dtype=k2_dtype)
        started = time.perf_counter()
        optimized = optimized_plans(mlp, alphas, payload)
        optimized_plan_time = time.perf_counter() - started
        plans = {
            CONTROL.name: control_plans(mlp, alphas),
            OPTIMIZED.name: optimized,
        }

        ordered = CANDIDATES if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(CANDIDATES))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_diagnostics = structural_prediction(
                    mlp,
                    pointsets[candidate.n_samples],
                    k2,
                    alphas,
                    {},
                    plans[candidate.name],
                    identity,
                    candidate,
                )
            uses_merge = candidate.selection != "none"
            flops_used = k2_flops + int(context.flops_used)
            residual = k2_residual + float(context.residual_wall_time_s or 0.0)
            plan_time = optimized_plan_time if uses_merge else 0.0
            if uses_merge:
                flops_used += merge_flops
                residual += merge_residual + plan_time
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
    prediction_arrays = {name: np.stack(value) for name, value in predictions.items()}
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
        "combo_final_mse",
        "combo_all_layer_mse",
        "combo_adjusted_score",
        "combo_flop_only_adjusted_score",
        "flops",
        "effective_compute",
    )
    crossfit = {}
    for candidate in CANDIDATES:
        left = parity["0"][candidate.name]
        right = parity["1"][candidate.name]
        crossfit[candidate.name] = {
            field: 0.5 * (left[field]["mean"] + right[field]["mean"])
            for field in fields
        }
        crossfit[candidate.name]["lambda_mean"] = 0.5 * (
            left["lambda"] + right["lambda"]
        )

    result = {
        "schema_version": "1.0",
        "experiment": "hope_merge_optimized_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "budget": args.budget,
            "static_merge_counts": STATIC_MERGE_COUNTS,
            "anchors_per_victim": ANCHORS_PER_VICTIM,
            "top_pairs_per_layer": TOP_PAIRS_PER_LAYER,
            "k2_dtype": args.k2_dtype,
            "flopscope_version": version("flopscope"),
            "whestbench_version": version("whestbench"),
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
        },
        "summary": {
            "parity_crossfit": parity,
            "crossfit_mean": crossfit,
            "rankings": {
                field: sorted(crossfit, key=lambda name: crossfit[name][field])
                for field in (
                    "combo_final_mse",
                    "combo_adjusted_score",
                    "combo_flop_only_adjusted_score",
                )
            },
        },
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
    stem = f"hope_merge_optimized_{args.k2_dtype}_limit{args.limit_mlps}"
    json_path = RESULTS / f"{stem}.json"
    cache_path = RESULTS / f"{stem}.npz"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        cache_path,
        mlp_ids=ids_array,
        targets=targets_array,
        k2=k2_array,
        **{f"prediction__{name}": value for name, value in prediction_arrays.items()},
    )
    return json_path, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--k2-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    output = run(args)
    print(f"wrote {output[0].relative_to(ROOT)} and {output[1].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
