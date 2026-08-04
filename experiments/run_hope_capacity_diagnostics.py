"""Diagnose HOPE-style pruning scores and functional mergeability."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import whestbench
from run_baselines import DATASET, RESULTS, REVISION, as_mlp
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from run_structural_tail_scout import antithetic_sobol_samples, sobol_base_integers

SEED_OFFSET = 8_675_309
EPS = 1e-30


def normal_cdf(values: np.ndarray) -> np.ndarray:
    scaled = np.ravel(values) / math.sqrt(2.0)
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value))) for value in scaled),
        dtype=np.float64,
        count=scaled.size,
    )
    return result.reshape(values.shape)


def normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def numpy_moments(
    weights: list[np.ndarray],
) -> dict[str, np.ndarray]:
    width = weights[0].shape[0]
    mu = np.zeros(width, dtype=np.float64)
    cov = np.eye(width, dtype=np.float64)
    log_scale = 0.0
    means = []
    alphas = []
    second_moments = []
    gains = []

    for weight in weights:
        max_var = float(np.max(np.diag(cov)))
        if max_var > 1e100:
            scale = math.sqrt(max_var)
            mu /= scale
            cov /= scale * scale
            log_scale += math.log(scale)

        mu_pre = weight.T @ mu
        cov_pre = weight.T @ cov @ weight
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = normal_pdf(alpha)
        gain = normal_cdf(alpha)
        mu = mu_pre * gain + sigma * phi
        second = (mu_pre * mu_pre + var_pre) * gain + mu_pre * sigma * phi
        var_post = np.maximum(second - mu * mu, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)

        layer_scale = math.exp(log_scale)
        means.append(mu * layer_scale)
        alphas.append(alpha)
        # A common layer scale does not affect within-layer pruning ranks.
        second_moments.append(second)
        gains.append(gain)

    second_array = np.stack(second_moments)
    gain_array = np.stack(gains)
    hope = np.sqrt(np.maximum(second_array, 0.0))
    for layer in range(len(weights) - 1):
        hope[layer] *= np.linalg.norm(weights[layer + 1], axis=1)

    downstream = np.empty_like(second_array)
    path_energy = np.ones(width, dtype=np.float64)
    downstream[-1] = np.sqrt(np.maximum(second_array[-1], 0.0))
    for layer in range(len(weights) - 2, -1, -1):
        weighted = gain_array[layer + 1] ** 2 * path_energy
        path_energy = (weights[layer + 1] ** 2) @ weighted
        downstream[layer] = np.sqrt(
            np.maximum(second_array[layer] * path_energy, 0.0)
        )

    return {
        "means": np.stack(means),
        "alphas": np.stack(alphas),
        "second_moments": second_array,
        "gains": gain_array,
        "hope": hope,
        "downstream": downstream,
    }


def selected_neurons(
    alpha: np.ndarray,
    threshold: float,
    hope: np.ndarray,
    downstream: np.ndarray,
) -> list[int]:
    eligible = np.flatnonzero(alpha > -threshold)
    if eligible.size <= 8:
        return eligible.tolist()

    selected: set[int] = set()
    for values in (alpha, hope, downstream):
        order = eligible[np.argsort(values[eligible])]
        selected.update(int(index) for index in order[:2])
    down_order = eligible[np.argsort(downstream[eligible])]
    for fraction in (0.25, 0.5):
        selected.add(int(down_order[min(len(down_order) - 1, int(fraction * len(down_order)))]))
    return sorted(selected)


def forward_activations(
    samples: np.ndarray,
    weights: list[np.ndarray],
) -> list[np.ndarray]:
    activations = []
    current = samples
    for weight in weights:
        current = np.maximum(current @ weight, 0.0)
        activations.append(current)
    return activations


def leave_one_effect(
    activation: np.ndarray,
    neuron: int,
    downstream_weights: list[np.ndarray],
    baseline_final_mean: np.ndarray,
) -> float:
    current = activation.copy()
    current[:, neuron] = 0.0
    for weight in downstream_weights:
        current = np.maximum(current @ weight, 0.0)
    delta = current.mean(axis=0, dtype=np.float64) - baseline_final_mean
    return float(np.mean(delta * delta))


def merge_layer_metrics(
    activation: np.ndarray,
    outgoing: np.ndarray,
    alive: np.ndarray,
) -> dict[str, Any]:
    activation = activation[:, alive].astype(np.float64)
    outgoing = outgoing[alive].astype(np.float64)
    count = activation.shape[1]
    if count < 2:
        return {"alive": count, "pairs": 0}

    kernel = activation.T @ activation / activation.shape[0]
    diagonal = np.maximum(np.diag(kernel), EPS)
    activation_cosine = kernel / np.sqrt(np.outer(diagonal, diagonal))

    centered = activation - activation.mean(axis=0, keepdims=True)
    centered_gram = centered.T @ centered
    centered_diag = np.maximum(np.diag(centered_gram), EPS)
    centered_cosine = centered_gram / np.sqrt(
        np.outer(centered_diag, centered_diag)
    )

    outgoing_gram = outgoing @ outgoing.T
    outgoing_energy = np.maximum(np.diag(outgoing_gram), EPS)
    outgoing_cosine = outgoing_gram / np.sqrt(
        np.outer(outgoing_energy, outgoing_energy)
    )
    functional_cosine = activation_cosine * outgoing_cosine

    residual_i_to_j = np.maximum(
        diagonal[:, None] - kernel * kernel / diagonal[None, :], 0.0
    ) * outgoing_energy[:, None]
    residual_j_to_i = residual_i_to_j.T
    total_energy = (
        diagonal[:, None] * outgoing_energy[:, None]
        + diagonal[None, :] * outgoing_energy[None, :]
        + 2.0 * kernel * outgoing_gram
    )
    relative_merge = np.sqrt(
        np.minimum(residual_i_to_j, residual_j_to_i)
        / np.maximum(total_energy, EPS)
    )

    upper = np.triu_indices(count, 1)
    activation_values = activation_cosine[upper]
    centered_values = centered_cosine[upper]
    functional_values = functional_cosine[upper]
    merge_values = relative_merge[upper]
    best = int(np.argmin(merge_values))
    return {
        "alive": count,
        "pairs": int(merge_values.size),
        "max_activation_cosine": float(np.max(activation_values)),
        "max_centered_cosine": float(np.max(centered_values)),
        "max_functional_cosine": float(np.max(functional_values)),
        "max_abs_functional_cosine": float(np.max(np.abs(functional_values))),
        "min_relative_merge_rmse": float(merge_values[best]),
        "best_pair": [int(alive[upper[0][best]]), int(alive[upper[1][best]])],
        "activation_pairs_above_090": int(np.sum(activation_values >= 0.90)),
        "activation_pairs_above_095": int(np.sum(activation_values >= 0.95)),
        "functional_pairs_above_090": int(np.sum(functional_values >= 0.90)),
        "merge_pairs_below_001": int(np.sum(merge_values <= 0.01)),
        "merge_pairs_below_005": int(np.sum(merge_values <= 0.05)),
        "merge_pairs_below_010": int(np.sum(merge_values <= 0.10)),
    }


def summarize_leave_one(records: list[dict[str, Any]]) -> dict[str, Any]:
    effects = np.asarray([row["effect_mse"] for row in records])
    predictors = ("alpha", "tail_probability", "second_moment", "hope", "downstream")
    output = {}
    for predictor in predictors:
        values = np.asarray([row[predictor] for row in records])
        output[predictor] = {
            "spearman": correlation(rankdata(values), rankdata(effects)),
            "log_pearson": correlation(
                np.log(np.maximum(np.abs(values), EPS)),
                np.log(np.maximum(effects, EPS)),
            ),
        }
    return output


def summarize_merge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "max_activation_cosine",
        "max_centered_cosine",
        "max_functional_cosine",
        "max_abs_functional_cosine",
        "min_relative_merge_rmse",
    )
    count_fields = (
        "pairs",
        "activation_pairs_above_090",
        "activation_pairs_above_095",
        "functional_pairs_above_090",
        "merge_pairs_below_001",
        "merge_pairs_below_005",
        "merge_pairs_below_010",
    )
    output: dict[str, Any] = {}
    for field in numeric_fields:
        values = np.asarray([row[field] for row in rows if field in row])
        output[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    for field in count_fields:
        output[field] = int(sum(int(row.get(field, 0)) for row in rows))
    return output


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    width = as_mlp(rows[0]).width
    base = sobol_base_integers(
        args.samples,
        width,
        args.sobol_direction_file,
        "numpy",
    )
    samples = np.asarray(
        antithetic_sobol_samples(base, 0, SEED_OFFSET, "numpy"),
        dtype=np.float32,
    )

    leave_one_records: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        weights = [np.asarray(weight, dtype=np.float32) for weight in mlp.weights]
        moments = numpy_moments(weights)
        activations = forward_activations(samples, weights)
        baseline_mean = activations[-1].mean(axis=0, dtype=np.float64)
        tail_start = mlp.depth - len(P_THRESHOLDS)

        for layer in range(mlp.depth - 6, mlp.depth - 2):
            threshold = P_THRESHOLDS[layer - tail_start]
            chosen = selected_neurons(
                moments["alphas"][layer],
                threshold,
                moments["hope"][layer],
                moments["downstream"][layer],
            )
            for neuron in chosen:
                alpha = float(moments["alphas"][layer, neuron])
                leave_one_records.append({
                    "mlp_id": int(row["mlp_id"]),
                    "layer": layer + 1,
                    "neuron": neuron,
                    "alpha": alpha,
                    "tail_probability": float(normal_cdf(np.asarray(alpha))),
                    "second_moment": float(
                        moments["second_moments"][layer, neuron]
                    ),
                    "hope": float(moments["hope"][layer, neuron]),
                    "downstream": float(moments["downstream"][layer, neuron]),
                    "effect_mse": leave_one_effect(
                        activations[layer],
                        neuron,
                        weights[layer + 1 :],
                        baseline_mean,
                    ),
                })

        for layer in range(tail_start, mlp.depth - 2):
            threshold = P_THRESHOLDS[layer - tail_start]
            alive = np.flatnonzero(moments["alphas"][layer] > -threshold)
            sample_energy = np.mean(
                activations[layer].astype(np.float64) ** 2,
                axis=0,
            )
            firing_rate = np.mean(activations[layer] > 0.0, axis=0)
            median_energy = float(np.median(sample_energy[alive]))
            meaningful = (
                (sample_energy >= max(EPS, 1e-3 * median_energy))
                & (firing_rate >= 0.01)
            )
            alive = alive[meaningful[alive]]
            metrics = merge_layer_metrics(
                activations[layer],
                weights[layer + 1],
                alive,
            )
            metrics.update({"mlp_id": int(row["mlp_id"]), "layer": layer + 1})
            merge_records.append(metrics)

        print(f"{row_number}/{len(rows)} {row['mlp_name']} completed", flush=True)

    result = {
        "schema_version": "1.0",
        "experiment": "hope_capacity_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "samples": args.samples,
            "leave_one_layers": [27, 28, 29, 30],
            "merge_layers": list(range(17, 31)),
        },
        "summary": {
            "leave_one": summarize_leave_one(leave_one_records),
            "merge": summarize_merge(merge_records),
        },
        "leave_one_records": leave_one_records,
        "merge_records": merge_records,
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"hope_capacity_diagnostics_limit{args.limit_mlps}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument(
        "--sobol-direction-file",
        type=Path,
        default=DEFAULT_SOBOL_DIRECTION_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be a positive even integer")
    output = run(args)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
