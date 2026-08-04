"""Test pilot-verified dead pruning, exact sparse products, and larger QMC sets."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, SCORE_FLOOR, as_mlp
from run_deep_structural_tail_scout import gaussian_classifier
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_structural_tail_scout import (
    antithetic_sobol_samples,
    sobol_base_integers,
)
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
FIXED_POINTSET_SEED = 0
SEED_OFFSET = 8_675_309
F_THRESHOLDS = (3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0)
K_THRESHOLDS = (4.75, 4.5, 4.25, 4.0) + F_THRESHOLDS
M_THRESHOLDS = (5.75, 5.5, 5.25, 5.0) + K_THRESHOLDS
Q_THRESHOLDS = tuple(value - 0.75 for value in M_THRESHOLDS[:4]) + M_THRESHOLDS[4:]
P_THRESHOLDS = tuple(value - 0.5 for value in M_THRESHOLDS[:-2]) + M_THRESHOLDS[-2:]
R_THRESHOLDS = tuple(value - 0.75 for value in M_THRESHOLDS[:-2]) + M_THRESHOLDS[-2:]
SPARSE_BUCKET_CAPS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Candidate:
    name: str
    n_samples: int
    thresholds: tuple[float, ...]
    middle_mode: str
    sparse: bool
    sparse_method: str = "gathered"
    pilot_samples: int = 0
    cold_target: float = 3.0
    min_hot: int = 96
    row_chunk: int = 100_000


KERNEL_CANDIDATES = (
    Candidate("Q_dense32_control", 32_768, Q_THRESHOLDS, "q", False),
    Candidate("R_dead_dense32", 32_768, Q_THRESHOLDS, "dead_only", False),
    Candidate("T_dead_m050_dense32", 32_768, P_THRESHOLDS, "dead_only", False),
    Candidate(
        "U_pilot512_m050_dense32",
        32_768,
        P_THRESHOLDS,
        "dead_only",
        False,
        pilot_samples=512,
    ),
    Candidate(
        "V_pilot1024_m075_dense32",
        32_768,
        R_THRESHOLDS,
        "dead_only",
        False,
        pilot_samples=1_024,
    ),
)


def sample_candidates(thresholds: tuple[float, ...] = P_THRESHOLDS) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            f"S_dead_m050_dense{n_samples}",
            n_samples,
            thresholds,
            "dead_only",
            False,
        )
        for n_samples in (32_768, 49_152, 65_536, 84_992)
    )


def floor_candidates(thresholds: tuple[float, ...] = P_THRESHOLDS) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            f"F_dead_m050_dense{n_samples}",
            n_samples,
            thresholds,
            "dead_only",
            False,
        )
        for n_samples in (32_768, 33_280, 33_792, 34_816)
    )


def candidates_for_stage(stage: str) -> tuple[Candidate, ...]:
    if stage == "kernel":
        return KERNEL_CANDIDATES
    if stage == "samples":
        return sample_candidates()
    if stage == "floor":
        return floor_candidates()
    raise ValueError(f"unknown stage {stage!r}")


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def scalar_classes(
    alpha: np.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int]]:
    active = np.flatnonzero(alpha >= threshold).tolist()
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    uncertain = np.flatnonzero(np.abs(alpha) < threshold).tolist()
    return active, dead, uncertain


def dead_only_classes(
    alpha: np.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int]]:
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    dead_mask = np.zeros(alpha.size, dtype=bool)
    dead_mask[dead] = True
    return [], dead, np.flatnonzero(~dead_mask).tolist()


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    result = parts[0]
    for part in parts[1:]:
        result = fnp.add(result, part)
    return result


def choose_cold_columns(
    x: fnp.ndarray, cold_target: float, min_hot: int
) -> tuple[list[int], list[int], float]:
    rates = [float(value) for value in fnp.mean(x > 0.0, axis=0).tolist()]
    order = sorted(range(len(rates)), key=rates.__getitem__)
    max_cold = max(0, len(order) - min_hot)
    cold_count = 0
    expected_nonzeros = 0.0
    for index in order[:max_cold]:
        next_expected = expected_nonzeros + rates[index]
        if next_expected > cold_target:
            break
        expected_nonzeros = next_expected
        cold_count += 1
    return order[:cold_count], order[cold_count:], expected_nonzeros


def _gathered_bucket_product(
    x_cold: fnp.ndarray,
    w_cold: fnp.ndarray,
    cap: int,
) -> fnp.ndarray:
    support_order = fnp.argsort(x_cold <= 0.0, axis=1)
    indices = support_order[:, :cap]
    values = fnp.take_along_axis(x_cold, indices, axis=1)
    selected_weights = w_cold[indices, :]
    return fnp.einsum("rk,rko->ro", values, selected_weights)


def exact_sparse_matmul(
    x: fnp.ndarray,
    w: fnp.ndarray,
    cold_target: float,
    min_hot: int,
    row_chunk: int,
) -> tuple[fnp.ndarray, dict[str, float | int]]:
    """Multiply exactly while skipping sampled zeros in a cold column prefix."""
    cold, hot, expected_nonzeros = choose_cold_columns(x, cold_target, min_hot)
    if len(cold) < 8:
        return x @ w, {
            "prefix_width": 0,
            "expected_cold_nonzeros": 0.0,
            "zero_rows": 0,
            "dense_rows": int(x.shape[0]),
        }

    x_cold = x[:, cold]
    counts = fnp.sum(x_cold > 0.0, axis=1)
    row_order = fnp.argsort(counts)
    sorted_counts = counts[row_order]
    x_sorted = x[row_order]
    hot_product = x_sorted[:, hot] @ w[hot, :]
    boundaries = {
        cap: int(fnp.searchsorted(sorted_counts, cap, side="right"))
        for cap in (0,) + SPARSE_BUCKET_CAPS
    }

    pieces = []
    start = 0
    for cap in (0,) + SPARSE_BUCKET_CAPS:
        stop = boundaries[cap]
        for chunk_start in range(start, stop, row_chunk):
            chunk_stop = min(chunk_start + row_chunk, stop)
            hot_part = hot_product[chunk_start:chunk_stop]
            if cap == 0:
                pieces.append(hot_part)
            else:
                cold_part = _gathered_bucket_product(
                    x_sorted[chunk_start:chunk_stop, :][:, cold],
                    w[cold, :],
                    cap,
                )
                pieces.append(fnp.add(hot_part, cold_part))
        start = stop

    for chunk_start in range(start, x.shape[0], row_chunk):
        chunk_stop = min(chunk_start + row_chunk, x.shape[0])
        cold_part = x_sorted[chunk_start:chunk_stop, :][:, cold] @ w[cold, :]
        pieces.append(fnp.add(hot_product[chunk_start:chunk_stop], cold_part))

    return fnp.concatenate(pieces, axis=0), {
        "prefix_width": len(cold),
        "expected_cold_nonzeros": expected_nonzeros,
        "zero_rows": boundaries[0],
        "dense_rows": int(x.shape[0]) - boundaries[8],
    }


def exact_zero_block_matmul(
    x: fnp.ndarray,
    w: fnp.ndarray,
    cold_target: float,
    min_hot: int,
) -> tuple[fnp.ndarray, dict[str, float | int]]:
    """Skip only the cold-prefix rows that are exactly all zero."""
    cold, hot, expected_nonzeros = choose_cold_columns(x, cold_target, min_hot)
    if len(cold) < 8:
        return x @ w, {
            "prefix_width": 0,
            "expected_cold_nonzeros": 0.0,
            "zero_rows": 0,
            "dense_rows": int(x.shape[0]),
        }

    counts = fnp.sum(x[:, cold] > 0.0, axis=1)
    row_order = fnp.argsort(counts)
    sorted_counts = counts[row_order]
    zero_rows = int(fnp.searchsorted(sorted_counts, 0, side="right"))
    x_sorted = x[row_order]
    hot_product = x_sorted[:, hot] @ w[hot, :]
    if zero_rows == x.shape[0]:
        output = hot_product
    elif zero_rows == 0:
        output = fnp.add(hot_product, x_sorted[:, cold] @ w[cold, :])
    else:
        cold_product = x_sorted[zero_rows:, :][:, cold] @ w[cold, :]
        output = fnp.concatenate(
            (
                hot_product[:zero_rows],
                fnp.add(hot_product[zero_rows:], cold_product),
            ),
            axis=0,
        )
    return output, {
        "prefix_width": len(cold),
        "expected_cold_nonzeros": expected_nonzeros,
        "zero_rows": zero_rows,
        "dense_rows": int(x.shape[0]) - zero_rows,
    }


def sparse_matmul(
    x: fnp.ndarray,
    w: fnp.ndarray,
    candidate: Candidate,
) -> tuple[fnp.ndarray, dict[str, float | int]]:
    if candidate.sparse_method == "zero_block":
        return exact_zero_block_matmul(
            x, w, candidate.cold_target, candidate.min_hot
        )
    if candidate.sparse_method == "gathered":
        return exact_sparse_matmul(
            x,
            w,
            candidate.cold_target,
            candidate.min_hot,
            candidate.row_chunk,
        )
    raise ValueError(f"unknown sparse method {candidate.sparse_method!r}")


def refine_dead_with_pilot(
    dead: list[int],
    pilot_z: fnp.ndarray,
    width: int,
) -> tuple[list[int], list[int], int]:
    if not dead:
        return [], list(range(width)), 0
    keep = np.asarray(fnp.all(pilot_z[:, dead] <= 0.0, axis=0), dtype=bool)
    refined_dead = np.asarray(dead, dtype=np.int64)[keep].tolist()
    dead_mask = np.zeros(width, dtype=bool)
    dead_mask[refined_dead] = True
    alive = np.flatnonzero(~dead_mask).tolist()
    return refined_dead, alive, int(len(dead) - len(refined_dead))


def structural_final_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    k2: fnp.ndarray,
    alphas: list[np.ndarray],
    identity: fnp.ndarray,
    candidate: Candidate,
) -> tuple[fnp.ndarray, dict[str, Any]]:
    """Propagate samples but only materialize the scored final sampled mean."""
    start = mlp.depth - len(candidate.thresholds)
    x = samples
    for w in mlp.weights[:start]:
        x = fnp.maximum(x @ w, 0.0)

    previous_alive: list[int] | None = None
    layer_diagnostics = {}
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        threshold = candidate.thresholds[offset]
        use_dead_only = candidate.middle_mode == "dead_only" or offset < 4
        if use_dead_only:
            active, dead, uncertain = dead_only_classes(alphas[layer], threshold)
        else:
            active, dead, uncertain = scalar_classes(alphas[layer], threshold)

        w = mlp.weights[layer]
        w_input = w if previous_alive is None else w[previous_alive, :]
        pilot_z = None
        demoted = 0
        if candidate.pilot_samples and use_dead_only:
            pilot_count = min(candidate.pilot_samples, x.shape[0])
            pilot_z = x[:pilot_count] @ w_input
            dead, uncertain, demoted = refine_dead_with_pilot(
                dead, pilot_z, mlp.width
            )
            active = []
        alive = active + uncertain
        if not alive:
            raise RuntimeError(f"all neurons classified dead at layer {layer + 1}")

        sparse_stats = {
            "prefix_width": 0,
            "expected_cold_nonzeros": 0.0,
            "zero_rows": 0,
            "dense_rows": int(x.shape[0]),
        }
        if pilot_z is not None:
            pilot_count = min(candidate.pilot_samples, x.shape[0])
            pilot_alive = pilot_z[:, alive]
            continuation = x[pilot_count:]
            if continuation.shape[0]:
                if candidate.sparse:
                    continuation_z, sparse_stats = sparse_matmul(
                        continuation, w_input[:, alive], candidate
                    )
                else:
                    continuation_z = continuation @ w_input[:, alive]
                z_alive = fnp.concatenate((pilot_alive, continuation_z), axis=0)
            else:
                z_alive = pilot_alive
        elif candidate.sparse:
            z_alive, sparse_stats = sparse_matmul(
                x, w_input[:, alive], candidate
            )
        else:
            z_alive = x @ w_input[:, alive]

        parts = []
        if active:
            parts.append(z_alive[:, : len(active)])
        if uncertain:
            parts.append(fnp.maximum(z_alive[:, len(active) :], 0.0))
        x = parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=1)
        previous_alive = alive
        layer_diagnostics[str(layer + 1)] = {
            "active": len(active),
            "dead": len(dead),
            "uncertain": len(uncertain),
            "pilot_demoted_dead": demoted,
            **sparse_stats,
        }

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = scalar_classes(
        alphas[layer1], candidate.thresholds[-2]
    )
    active2, dead2, uncertain2 = scalar_classes(
        alphas[layer2], candidate.thresholds[-1]
    )
    layer_diagnostics[str(layer1 + 1)] = {
        "active": len(active1), "dead": len(dead1), "uncertain": len(uncertain1)
    }
    layer_diagnostics[str(layer2 + 1)] = {
        "active": len(active2), "dead": len(dead2), "uncertain": len(uncertain2)
    }

    w1 = mlp.weights[layer1]
    w2 = mlp.weights[layer2]
    w1_input = w1 if previous_alive is None else w1[previous_alive, :]
    mean_x = fnp.mean(x, axis=0)
    mean1_parts = []
    h1_uncertain = None
    if active1:
        mean1_parts.append((mean_x @ w1_input[:, active1]) @ identity[active1, :])
    if uncertain1:
        h1_uncertain = fnp.maximum(x @ w1_input[:, uncertain1], 0.0)
        mean1_parts.append(
            fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :]
        )
    mean1 = _sum_parts(mean1_parts, mlp.width)

    mean2_parts = []
    if active2:
        mean2_parts.append((mean1 @ w2[:, active2]) @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            collapsed = w1_input[:, active1] @ w2[active1, :][:, uncertain2]
            z2_uncertain = x @ collapsed
        if uncertain1:
            contribution = h1_uncertain @ w2[uncertain1, :][:, uncertain2]
            z2_uncertain = (
                contribution
                if z2_uncertain is None
                else fnp.add(z2_uncertain, contribution)
            )
        if z2_uncertain is not None:
            mean2_parts.append(
                fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0)
                @ identity[uncertain2, :]
            )
    mean2 = _sum_parts(mean2_parts, mlp.width)
    prediction = fnp.concatenate((k2[:-1], mean2[None, :]), axis=0)
    return prediction, layer_diagnostics


def fit_lambda(
    ids: np.ndarray,
    k2: np.ndarray,
    sampled: np.ndarray,
    targets: np.ndarray,
    parity: int,
) -> float:
    selected = ids % 2 == parity
    delta = sampled[selected, -1] - k2[selected, -1]
    baseline_error = k2[selected, -1] - targets[selected, -1]
    denominator = float(np.vdot(delta, delta))
    value = -float(np.vdot(delta, baseline_error)) / denominator if denominator else 0.0
    return min(1.0, max(0.0, value))


def summarize_candidate(
    candidate: Candidate,
    ids: np.ndarray,
    k2: np.ndarray,
    sampled: np.ndarray,
    targets: np.ndarray,
    compute: list[dict[str, float | int]],
    diagnostics: list[dict[str, Any]],
    parity: int,
    budget: int,
) -> dict[str, Any]:
    evaluation = ids % 2 != parity
    lambda_ = fit_lambda(ids, k2, sampled, targets, parity)
    combo = k2[evaluation] + lambda_ * (sampled[evaluation] - k2[evaluation])
    target = targets[evaluation]
    final_per_mlp = ((combo[:, -1] - target[:, -1]) ** 2).mean(axis=1)
    all_per_mlp = ((combo - target) ** 2).mean(axis=(1, 2))
    selected_compute = [row for row, keep in zip(compute, evaluation) if keep]
    flops_used = np.asarray([row["flops"] for row in selected_compute], dtype=np.float64)
    effective = np.asarray(
        [row["effective_compute"] for row in selected_compute], dtype=np.float64
    )
    selected_diagnostics = [row for row, keep in zip(diagnostics, evaluation) if keep]
    layer_summary = {}
    for layer in selected_diagnostics[0]:
        fields = selected_diagnostics[0][layer]
        layer_summary[layer] = {
            field: aggregate(row[layer][field] for row in selected_diagnostics)
            for field in fields
        }
    return {
        "configuration": asdict(candidate),
        "lambda": lambda_,
        "combo_final_mse": aggregate(final_per_mlp),
        "combo_all_layer_mse": aggregate(all_per_mlp),
        "combo_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, effective / budget)
        ),
        "combo_flop_only_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, flops_used / budget)
        ),
        "flops": aggregate(flops_used),
        "effective_compute": aggregate(effective),
        "residual_wall_time_s": aggregate(
            row["residual_wall_time_s"] for row in selected_compute
        ),
        "runtime_s": aggregate(row["runtime_s"] for row in selected_compute),
        "budget_exceeded_fraction": float(np.mean(effective > budget)),
        "layer_diagnostics": layer_summary,
    }


def build_pointsets(
    candidates: tuple[Candidate, ...], width: int, direction_file: Path
) -> dict[int, fnp.ndarray]:
    output = {}
    for n_samples in sorted({candidate.n_samples for candidate in candidates}):
        base = sobol_base_integers(n_samples, width, direction_file, "numpy")
        output[n_samples] = antithetic_sobol_samples(
            base, FIXED_POINTSET_SEED, SEED_OFFSET, "numpy"
        )
        print(f"built nested Sobol n={n_samples}", flush=True)
    return output


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    candidates = candidates_for_stage(args.stage)
    if args.max_samples is not None:
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.n_samples <= args.max_samples
        )
        if not candidates:
            raise ValueError("--max-samples removed every candidate")
    width = as_mlp(rows[0]).width
    identity = fnp.array(np.eye(width, dtype=np.float32))
    pointsets = build_pointsets(candidates, width, args.sobol_direction_file)

    ids = []
    targets = []
    k2_predictions = []
    predictions: dict[str, list[np.ndarray]] = {c.name: [] for c in candidates}
    compute: dict[str, list[dict[str, float | int]]] = {c.name: [] for c in candidates}
    diagnostics: dict[str, list[dict[str, Any]]] = {c.name: [] for c in candidates}

    for row_index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        mlp_id = int(row["mlp_id"])
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
            k2, alphas = gaussian_classifier(mlp)
        k2_residual = float(k2_context.residual_wall_time_s or 0.0)
        ordered = candidates if (mlp_id // 2) % 2 == 0 else tuple(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_diagnostics = structural_final_prediction(
                    mlp,
                    pointsets[candidate.n_samples],
                    k2,
                    alphas,
                    identity,
                    candidate,
                )
            runtime_s = time.perf_counter() - started
            residual_s = k2_residual + float(context.residual_wall_time_s or 0.0)
            flops_used = int(k2_context.flops_used) + int(context.flops_used)
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            diagnostics[candidate.name].append(candidate_diagnostics)
            compute[candidate.name].append({
                "flops": flops_used,
                "residual_wall_time_s": residual_s,
                "effective_compute": flops_used + LAMBDA_FLOPS_PER_SECOND * residual_s,
                "runtime_s": runtime_s,
            })

        ids.append(mlp_id)
        targets.append(np.asarray(row["all_layer_means"], dtype=np.float64))
        k2_predictions.append(np.asarray(k2, dtype=np.float64))
        print(f"{row_index}/{len(rows)} {row['mlp_name']} completed", flush=True)

    ids_array = np.asarray(ids, dtype=np.int64)
    targets_array = np.stack(targets)
    k2_array = np.stack(k2_predictions)
    prediction_arrays = {name: np.stack(value) for name, value in predictions.items()}
    parity_summaries = {
        str(parity): {
            candidate.name: summarize_candidate(
                candidate,
                ids_array,
                k2_array,
                prediction_arrays[candidate.name],
                targets_array,
                compute[candidate.name],
                diagnostics[candidate.name],
                parity,
                args.budget,
            )
            for candidate in candidates
        }
        for parity in (0, 1)
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
    for candidate in candidates:
        p0 = parity_summaries["0"][candidate.name]
        p1 = parity_summaries["1"][candidate.name]
        crossfit[candidate.name] = {
            field: 0.5 * (p0[field]["mean"] + p1[field]["mean"])
            for field in fields
        }
        crossfit[candidate.name]["lambda_mean"] = 0.5 * (
            p0["lambda"] + p1["lambda"]
        )

    result = {
        "schema_version": "1.0",
        "experiment": "sparse_pilot_reinvestment_scout",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "budget": args.budget,
            "balanced_order": True,
            "final_only_output": True,
            "fixed_pointset_seed": FIXED_POINTSET_SEED,
            "seed_offset": SEED_OFFSET,
            "candidates": [asdict(candidate) for candidate in candidates],
        },
        "summary": {
            "parity_crossfit": parity_summaries,
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
                "compute": {c.name: compute[c.name][index] for c in candidates},
                "diagnostics": {
                    c.name: diagnostics[c.name][index] for c in candidates
                },
            }
            for index, mlp_id in enumerate(ids)
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    stem = f"sparse_pilot_reinvestment_{args.stage}_limit{args.limit_mlps}"
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
    parser.add_argument(
        "--stage", choices=("kernel", "samples", "floor"), required=True
    )
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    if not args.sobol_direction_file.exists():
        raise SystemExit(f"Sobol direction file not found: {args.sobol_direction_file}")
    json_path, cache_path = run(args)
    print(f"wrote {json_path.relative_to(ROOT)} and {cache_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
