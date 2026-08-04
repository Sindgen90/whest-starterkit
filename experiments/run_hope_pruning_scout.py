"""Test HOPE-style structured pruning and mean-preserving corrections.

The rank stage compares Gaussian tail, one-step HOPE capacity, and diagonal
downstream path-energy scores. The correction stage applies a constant omitted
activation mean to the best ranker selected by the caller.
"""
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
from run_deep_structural_tail_scout import K2_COEFFICIENT
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from run_structural_tail_scout import antithetic_sobol_samples, sobol_base_integers
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 32_768
SEED_OFFSET = 8_675_309
FIXED_POINTSET_SEED = 0
_COV_RESCALE_THRESHOLD = 1e100


@dataclass(frozen=True)
class Candidate:
    name: str
    score_mode: str = "none"
    extra_per_layer: int = 0
    correction_mode: str = "none"
    correction_scale: float = 0.0


RANK_CANDIDATES = (
    Candidate("control_dead_m050"),
    Candidate("tail_extra08", "tail", 8),
    Candidate("hope_extra08", "hope", 8),
    Candidate("downstream_extra08", "downstream", 8),
    Candidate("tail_extra16", "tail", 16),
    Candidate("hope_extra16", "hope", 16),
    Candidate("downstream_extra16", "downstream", 16),
)


def correction_candidates(score_mode: str, extra: int) -> tuple[Candidate, ...]:
    stem = f"{score_mode}_extra{extra:02d}"
    return (
        Candidate("control_dead_m050"),
        Candidate(stem, score_mode, extra),
        Candidate(f"{stem}_mean_extra_s050", score_mode, extra, "extra", 0.5),
        Candidate(f"{stem}_mean_extra_s100", score_mode, extra, "extra", 1.0),
        Candidate(f"{stem}_mean_all_s050", score_mode, extra, "all", 0.5),
        Candidate(f"{stem}_mean_all_s100", score_mode, extra, "all", 1.0),
        Candidate(f"{stem}_linear_extra_s050", score_mode, extra, "linear_extra", 0.5),
        Candidate(f"{stem}_linear_extra_s100", score_mode, extra, "linear_extra", 1.0),
        Candidate(f"{stem}_linear_all_s050", score_mode, extra, "linear_all", 0.5),
        Candidate(f"{stem}_linear_all_s100", score_mode, extra, "linear_all", 1.0),
    )


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


def gaussian_pruning_stats(
    mlp: Any,
) -> tuple[fnp.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, int, float]:
    """Compute K2 plus tail/HOPE/downstream ranks in separately metered phases."""
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
        mu = fnp.zeros(mlp.width)
        cov = fnp.eye(mlp.width)
        log_scale = 0.0
        rows: list[fnp.ndarray] = []
        alphas: list[fnp.ndarray] = []
        second_moments: list[fnp.ndarray] = []
        gains: list[fnp.ndarray] = []
        for weight in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var = float(fnp.max(cov_diag))
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
            gain64 = fnp.asarray(gain, dtype=fnp.float64)
            cov = fnp.multiply(fnp.outer(gain64, gain64), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            rows.append(mu * float(fnp.exp(log_scale)))
            alphas.append(alpha)
            # A common layer scale does not affect within-layer ranking.
            second_moments.append(second)
            gains.append(gain)

        k2 = fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT)
        classification_values = np.asarray(
            fnp.stack(
                (fnp.stack(alphas, axis=0), fnp.stack(gains, axis=0)), axis=0
            ),
            dtype=np.float64,
        )

    with flops.BudgetContext(flop_budget=10**15, quiet=True) as score_context:
        hope: list[fnp.ndarray] = []
        for layer in range(mlp.depth):
            score = fnp.sqrt(fnp.maximum(second_moments[layer], 0.0))
            if layer + 1 < mlp.depth:
                outgoing_norm = fnp.sqrt(
                    fnp.sum(mlp.weights[layer + 1] ** 2, axis=1)
                )
                score = score * outgoing_norm
            hope.append(score)

        downstream: list[fnp.ndarray | None] = [None] * mlp.depth
        path_energy = fnp.ones(mlp.width)
        downstream[-1] = fnp.sqrt(fnp.maximum(second_moments[-1], 0.0))
        for layer in range(mlp.depth - 2, -1, -1):
            weighted = gains[layer + 1] ** 2 * path_energy
            path_energy = (mlp.weights[layer + 1] ** 2) @ weighted
            downstream[layer] = fnp.sqrt(
                fnp.maximum(second_moments[layer] * path_energy, 0.0)
            )
        score_values = np.asarray(
            fnp.stack(
                (
                    fnp.stack(hope, axis=0),
                    fnp.stack(downstream, axis=0),
                ),
                axis=0,
            ),
            dtype=np.float64,
        )

    return (
        k2,
        classification_values[0],
        classification_values[1],
        score_values,
        int(k2_context.flops_used),
        float(k2_context.residual_wall_time_s or 0.0),
        int(score_context.flops_used),
        float(score_context.residual_wall_time_s or 0.0),
    )


def choose_neurons(
    alpha: np.ndarray,
    threshold: float,
    score: np.ndarray | None,
    extra: int,
) -> tuple[list[int], list[int], list[int]]:
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    alive = np.flatnonzero(alpha > -threshold)
    if score is None or extra <= 0:
        return dead, [], alive.tolist()
    count = min(extra, max(0, alive.size - 1))
    order = alive[np.argsort(score[alive], kind="stable")]
    additionally_pruned = order[:count].tolist()
    kept = order[count:].tolist()
    # Sorted indices make remote-array gathers deterministic and easier to inspect.
    return dead, sorted(additionally_pruned), sorted(kept)


def candidate_scores(
    candidate: Candidate,
    alpha: np.ndarray,
    score_values: np.ndarray,
    layer: int,
) -> np.ndarray | None:
    if candidate.score_mode == "none":
        return None
    if candidate.score_mode == "tail":
        return alpha[layer]
    if candidate.score_mode == "hope":
        return score_values[0, layer]
    if candidate.score_mode == "downstream":
        return score_values[1, layer]
    raise ValueError(f"unknown score mode {candidate.score_mode!r}")


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    total = parts[0]
    for part in parts[1:]:
        total = fnp.add(total, part)
    return total


def materialized_correction_biases(
    mlp: Any,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    score_values: np.ndarray,
    candidate: Candidate,
    start: int,
) -> fnp.ndarray | None:
    """Build and detach tiny correction vectors before sample propagation.

    Detaching the 14x256 table prevents every sampled layer from retaining the
    full K2 computation graph through its pending bias.
    """
    if candidate.correction_mode not in ("extra", "all"):
        return None
    biases = []
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        score = candidate_scores(candidate, alphas, score_values, layer)
        dead, extra, _ = choose_neurons(
            alphas[layer], P_THRESHOLDS[offset], score, candidate.extra_per_layer
        )
        corrected = extra if candidate.correction_mode == "extra" else dead + extra
        if corrected:
            biases.append(k2[layer, corrected] @ mlp.weights[layer + 1][corrected, :])
        else:
            biases.append(fnp.zeros(mlp.width, dtype=fnp.float32))
    values = np.asarray(fnp.stack(biases, axis=0), dtype=np.float32)
    return fnp.array(values)


def linear_response_correction(
    mlp: Any,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    gains: np.ndarray,
    score_values: np.ndarray,
    candidate: Candidate,
    start: int,
) -> fnp.ndarray | None:
    """Propagate all omitted means in one diagonal-Jacobian recurrence."""
    if candidate.correction_mode not in ("linear_extra", "linear_all"):
        return None
    delta = fnp.zeros(mlp.width, dtype=fnp.float32)
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        score = candidate_scores(candidate, alphas, score_values, layer)
        dead, extra, _ = choose_neurons(
            alphas[layer], P_THRESHOLDS[offset], score, candidate.extra_per_layer
        )
        corrected = (
            extra if candidate.correction_mode == "linear_extra" else dead + extra
        )
        next_delta = delta @ mlp.weights[layer + 1]
        if corrected:
            omitted = k2[layer, corrected] @ mlp.weights[layer + 1][corrected, :]
            next_delta = fnp.add(next_delta, omitted)
        delta = next_delta * fnp.array(gains[layer + 1].astype(np.float32))

    # Layers 31 and 32 are not additionally pruned, but they transmit the
    # accumulated response from layer 30 to the scored output.
    for layer in range(mlp.depth - 2, mlp.depth - 1):
        delta = (delta @ mlp.weights[layer + 1]) * fnp.array(
            gains[layer + 1].astype(np.float32)
        )
    return delta * candidate.correction_scale


def structural_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    gains: np.ndarray,
    score_values: np.ndarray,
    identity: fnp.ndarray,
    candidate: Candidate,
) -> tuple[fnp.ndarray, dict[str, Any]]:
    start = mlp.depth - len(P_THRESHOLDS)
    correction_biases = materialized_correction_biases(
        mlp, k2, alphas, score_values, candidate, start
    )
    final_correction = linear_response_correction(
        mlp, k2, alphas, gains, score_values, candidate, start
    )
    x = samples
    for weight in mlp.weights[:start]:
        x = fnp.maximum(x @ weight, 0.0)

    previous_alive: list[int] | None = None
    pending_bias: fnp.ndarray | None = None
    diagnostics: dict[str, Any] = {}
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        score = candidate_scores(candidate, alphas, score_values, layer)
        dead, extra, alive = choose_neurons(
            alphas[layer], P_THRESHOLDS[offset], score, candidate.extra_per_layer
        )
        if not alive:
            raise RuntimeError(f"all neurons classified dead at layer {layer + 1}")

        weight = mlp.weights[layer]
        weight_input = weight if previous_alive is None else weight[previous_alive, :]
        z_alive = x @ weight_input[:, alive]
        if pending_bias is not None:
            z_alive = fnp.add(z_alive, pending_bias[alive])
        x = fnp.maximum(z_alive, 0.0)
        previous_alive = alive

        corrected = (
            extra
            if candidate.correction_mode in ("extra", "linear_extra")
            else []
        )
        if candidate.correction_mode in ("all", "linear_all"):
            corrected = dead + extra
        if correction_biases is not None and corrected:
            pending_bias = correction_biases[offset] * candidate.correction_scale
        else:
            pending_bias = None
        diagnostics[str(layer + 1)] = {
            "dead": len(dead),
            "extra_pruned": len(extra),
            "alive": len(alive),
            "corrected": len(corrected),
        }

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    alpha1 = alphas[layer1]
    alpha2 = alphas[layer2]
    active1 = np.flatnonzero(alpha1 >= P_THRESHOLDS[-2]).tolist()
    dead1 = np.flatnonzero(alpha1 <= -P_THRESHOLDS[-2]).tolist()
    uncertain1 = np.flatnonzero(np.abs(alpha1) < P_THRESHOLDS[-2]).tolist()
    active2 = np.flatnonzero(alpha2 >= P_THRESHOLDS[-1]).tolist()
    dead2 = np.flatnonzero(alpha2 <= -P_THRESHOLDS[-1]).tolist()
    uncertain2 = np.flatnonzero(np.abs(alpha2) < P_THRESHOLDS[-1]).tolist()

    w1 = mlp.weights[layer1]
    w2 = mlp.weights[layer2]
    w1_input = w1 if previous_alive is None else w1[previous_alive, :]
    mean_x = fnp.mean(x, axis=0)
    mean1_parts: list[fnp.ndarray] = []
    h1_uncertain = None
    if active1:
        active_mean = mean_x @ w1_input[:, active1]
        if pending_bias is not None:
            active_mean = fnp.add(active_mean, pending_bias[active1])
        mean1_parts.append(active_mean @ identity[active1, :])
    if uncertain1:
        z1_uncertain = x @ w1_input[:, uncertain1]
        if pending_bias is not None:
            z1_uncertain = fnp.add(z1_uncertain, pending_bias[uncertain1])
        h1_uncertain = fnp.maximum(z1_uncertain, 0.0)
        mean1_parts.append(
            fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :]
        )
    mean1 = _sum_parts(mean1_parts, mlp.width)

    mean2_parts: list[fnp.ndarray] = []
    if active2:
        mean2_parts.append((mean1 @ w2[:, active2]) @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            collapsed = w1_input[:, active1] @ w2[active1, :][:, uncertain2]
            active_contribution = x @ collapsed
            if pending_bias is not None:
                active_bias = pending_bias[active1] @ w2[active1, :][:, uncertain2]
                active_contribution = fnp.add(active_contribution, active_bias)
            z2_uncertain = active_contribution
        if uncertain1:
            uncertain_contribution = h1_uncertain @ w2[uncertain1, :][:, uncertain2]
            z2_uncertain = (
                uncertain_contribution
                if z2_uncertain is None
                else fnp.add(z2_uncertain, uncertain_contribution)
            )
        if z2_uncertain is not None:
            mean2_parts.append(
                fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0)
                @ identity[uncertain2, :]
            )
    mean2 = _sum_parts(mean2_parts, mlp.width)
    if final_correction is not None:
        mean2 = fnp.add(mean2, final_correction)
    prediction = fnp.concatenate((k2[:-1], mean2[None, :]), axis=0)
    diagnostics[str(layer1 + 1)] = {
        "dead": len(dead1), "active": len(active1), "uncertain": len(uncertain1)
    }
    diagnostics[str(layer2 + 1)] = {
        "dead": len(dead2), "active": len(active2), "uncertain": len(uncertain2)
    }
    return prediction, diagnostics


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
    compute: list[dict[str, float]],
    diagnostics: list[dict[str, Any]],
    parity: int,
    budget: int,
) -> dict[str, Any]:
    evaluation = ids % 2 != parity
    lambda_ = fit_lambda(ids, k2, sampled, targets, parity)
    combo = k2[evaluation] + lambda_ * (sampled[evaluation] - k2[evaluation])
    target = targets[evaluation]
    final_mse = ((combo[:, -1] - target[:, -1]) ** 2).mean(axis=1)
    all_mse = ((combo - target) ** 2).mean(axis=(1, 2))
    selected_compute = [row for row, keep in zip(compute, evaluation) if keep]
    used = np.asarray([row["flops"] for row in selected_compute])
    effective = np.asarray([row["effective_compute"] for row in selected_compute])
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
        "combo_final_mse": aggregate(final_mse),
        "combo_all_layer_mse": aggregate(all_mse),
        "combo_adjusted_score": aggregate(
            final_mse * np.maximum(SCORE_FLOOR, effective / budget)
        ),
        "combo_flop_only_adjusted_score": aggregate(
            final_mse * np.maximum(SCORE_FLOOR, used / budget)
        ),
        "flops": aggregate(used),
        "effective_compute": aggregate(effective),
        "runtime_s": aggregate(row["runtime_s"] for row in selected_compute),
        "layer_diagnostics": layer_summary,
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    candidates = (
        RANK_CANDIDATES
        if args.stage == "rank"
        else correction_candidates(args.score_mode, args.extra_per_layer)
    )
    if args.candidate is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.name == args.candidate
        )
        if not candidates:
            raise ValueError(f"candidate not found: {args.candidate!r}")
    width = as_mlp(rows[0]).width
    base = sobol_base_integers(
        N_SAMPLES, width, args.sobol_direction_file, "numpy"
    )
    samples = antithetic_sobol_samples(base, FIXED_POINTSET_SEED, SEED_OFFSET, "numpy")
    identity = fnp.array(np.eye(width, dtype=np.float32))

    ids: list[int] = []
    targets: list[np.ndarray] = []
    k2_predictions: list[np.ndarray] = []
    predictions = {candidate.name: [] for candidate in candidates}
    compute = {candidate.name: [] for candidate in candidates}
    diagnostics = {candidate.name: [] for candidate in candidates}

    for row_index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        (
            k2,
            alphas,
            gains,
            scores,
            k2_flops,
            k2_residual,
            score_flops,
            score_residual,
        ) = gaussian_pruning_stats(mlp)
        ordered = candidates if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_diagnostics = structural_prediction(
                    mlp, samples, k2, alphas, gains, scores, identity, candidate
                )
            # Tail ranking reuses alpha from K2; only capacity/path scores need
            # the separate backward score pass in a submission.
            ranked = candidate.score_mode in ("hope", "downstream")
            residual = k2_residual + float(context.residual_wall_time_s or 0.0)
            flops_used = k2_flops + int(context.flops_used)
            if ranked:
                residual += score_residual
                flops_used += score_flops
            runtime = time.perf_counter() - started
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            diagnostics[candidate.name].append(candidate_diagnostics)
            compute[candidate.name].append({
                "flops": float(flops_used),
                "effective_compute": float(
                    flops_used + LAMBDA_FLOPS_PER_SECOND * residual
                ),
                "runtime_s": runtime,
            })

        ids.append(int(row["mlp_id"]))
        targets.append(np.asarray(row["all_layer_means"], dtype=np.float64))
        k2_predictions.append(np.asarray(k2, dtype=np.float64))
        print(f"{row_index}/{len(rows)} {row['mlp_name']} completed", flush=True)

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
            for candidate in candidates
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
    for candidate in candidates:
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
        "experiment": "hope_pruning_scout",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "samples": N_SAMPLES,
            "budget": args.budget,
            "balanced_order": True,
            "candidates": [asdict(candidate) for candidate in candidates],
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
                "compute": {candidate.name: compute[candidate.name][index] for candidate in candidates},
                "diagnostics": {candidate.name: diagnostics[candidate.name][index] for candidate in candidates},
            }
            for index, mlp_id in enumerate(ids_array)
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = f"{args.stage}_limit{args.limit_mlps}"
    if args.stage == "correction":
        suffix = f"correction_{args.score_mode}{args.extra_per_layer}_limit{args.limit_mlps}"
    if args.candidate is not None:
        suffix = f"{suffix}_{args.candidate}"
    json_path = RESULTS / f"hope_pruning_{suffix}.json"
    cache_path = RESULTS / f"hope_pruning_{suffix}.npz"
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
    parser.add_argument("--stage", choices=("rank", "correction"), required=True)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--score-mode", choices=("tail", "hope", "downstream"), default="tail")
    parser.add_argument("--extra-per-layer", type=int, default=8)
    parser.add_argument("--candidate")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
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
