"""Analyze adaptive sample selection and conditional shrinkage from QMC caches."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from run_baselines import BUDGET, RESULTS, SCORE_FLOOR
from run_structural_qmc_suite import aggregate, fit_lambda

ROOT = Path(__file__).resolve().parents[1]


def load_stage(stage: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_path = RESULTS / f"structural_qmc_{stage}_limit50.json"
    cache_path = RESULTS / f"structural_qmc_{stage}_limit50.npz"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    return metadata, arrays


def compute_arrays(metadata: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    flops_used = np.asarray(
        [row["compute"][name]["flops"] for row in metadata["per_mlp"]], dtype=np.float64
    )
    effective = np.asarray(
        [row["compute"][name]["effective_compute"] for row in metadata["per_mlp"]],
        dtype=np.float64,
    )
    return flops_used, effective


def combo_prediction(
    ids: np.ndarray,
    k2: np.ndarray,
    prediction: np.ndarray,
    targets: np.ndarray,
    calibration_parity: int,
) -> tuple[float, np.ndarray]:
    lambda_ = fit_lambda(ids, k2, prediction, targets, calibration_parity)
    return lambda_, k2 + lambda_ * (prediction - k2)


def selection_summary(
    ids: np.ndarray,
    targets: np.ndarray,
    low_combo: np.ndarray,
    high_combo: np.ndarray,
    choose_high: np.ndarray,
    low_flops: np.ndarray,
    high_flops: np.ndarray,
    low_effective: np.ndarray,
    high_effective: np.ndarray,
    calibration_parity: int,
    low_samples: int,
    high_samples: int,
    budget: int,
) -> dict[str, Any]:
    selected_prediction = np.where(choose_high[:, None, None], high_combo, low_combo)
    selected_flops = np.where(choose_high, high_flops, low_flops)
    selected_effective = np.where(choose_high, high_effective, low_effective)
    selected_samples = np.where(choose_high, high_samples, low_samples)
    evaluation = ids % 2 != calibration_parity
    error = (selected_prediction[evaluation] - targets[evaluation]) ** 2
    final_per_mlp = error[:, -1].mean(axis=1)
    return {
        "evaluation_mlps": int(evaluation.sum()),
        "high_fraction": float(np.mean(choose_high[evaluation])),
        "mean_samples": float(np.mean(selected_samples[evaluation])),
        "final_mse": aggregate(final_per_mlp),
        "all_layer_mse": aggregate(error.mean(axis=(1, 2))),
        "adjusted_score": aggregate(
            final_per_mlp
            * np.maximum(SCORE_FLOOR, selected_effective[evaluation] / budget)
        ),
        "flop_only_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, selected_flops[evaluation] / budget)
        ),
        "flops": aggregate(selected_flops[evaluation]),
        "effective_compute": aggregate(selected_effective[evaluation]),
    }


def controller_features(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, str]]:
    diagnostic_names = tuple(metadata["per_mlp"][0]["diagnostics"])
    features = {
        name: (
            np.asarray([row["diagnostics"][name] for row in metadata["per_mlp"]]),
            "low" if name.startswith("mean_abs_alpha") else "high",
        )
        for name in diagnostic_names
    }
    low = arrays["prediction__xor_n16384"]
    k2 = arrays["k2"]
    correction_rms = np.sqrt(np.mean((low[:, -1] - k2[:, -1]) ** 2, axis=1))
    k2_rms = np.sqrt(np.mean(k2[:, -1] ** 2, axis=1))
    features["k2_final_rms"] = (k2_rms, "high")
    features["qmc16_correction_rms"] = (correction_rms, "high")
    features["qmc16_relative_correction"] = (
        correction_rms / np.maximum(k2_rms, 1e-12),
        "high",
    )
    return features


def median_choice(
    values: np.ndarray, ids: np.ndarray, calibration_parity: int, direction: str
) -> tuple[float, np.ndarray]:
    threshold = float(np.median(values[ids % 2 == calibration_parity]))
    choose_high = values >= threshold if direction == "high" else values <= threshold
    return threshold, choose_high


def run_controller(args: argparse.Namespace) -> Path:
    metadata, arrays = load_stage("count")
    ids = arrays["mlp_ids"]
    targets = arrays["targets"]
    k2 = arrays["k2"]
    features = controller_features(metadata, arrays)
    candidate_names = ("xor_n16384", "xor_n28672", "xor_n32768")
    samples = {"xor_n16384": 16_384, "xor_n28672": 28_672, "xor_n32768": 32_768}

    parity_results = {}
    for parity in (0, 1):
        combos = {}
        lambdas = {}
        compute = {}
        for name in candidate_names:
            lambdas[name], combos[name] = combo_prediction(
                ids, k2, arrays[f"prediction__{name}"], targets, parity
            )
            compute[name] = compute_arrays(metadata, name)

        rules = {}
        low_name = "xor_n16384"
        for high_name in ("xor_n28672", "xor_n32768"):
            for feature_name, (values, direction) in features.items():
                threshold, choose_high = median_choice(values, ids, parity, direction)
                name = f"{low_name}_to_{high_name}__{feature_name}"
                rules[name] = {
                    "low": low_name,
                    "high": high_name,
                    "feature": feature_name,
                    "direction": direction,
                    "threshold": threshold,
                    **selection_summary(
                        ids,
                        targets,
                        combos[low_name],
                        combos[high_name],
                        choose_high,
                        compute[low_name][0],
                        compute[high_name][0],
                        compute[low_name][1],
                        compute[high_name][1],
                        parity,
                        samples[low_name],
                        samples[high_name],
                        args.budget,
                    ),
                }

            component_names = (
                "qmc16_relative_correction",
                "boundary_mass_2",
                "uncertain_fraction_2",
            )
            votes = np.zeros(ids.size, dtype=np.int64)
            thresholds = {}
            for component in component_names:
                values, direction = features[component]
                threshold, selected = median_choice(values, ids, parity, direction)
                thresholds[component] = threshold
                votes += selected
            choose_high = votes >= 2
            name = f"{low_name}_to_{high_name}__combined_vote"
            rules[name] = {
                "low": low_name,
                "high": high_name,
                "feature": "combined_vote",
                "direction": "at_least_two_high",
                "threshold": thresholds,
                **selection_summary(
                    ids,
                    targets,
                    combos[low_name],
                    combos[high_name],
                    choose_high,
                    compute[low_name][0],
                    compute[high_name][0],
                    compute[low_name][1],
                    compute[high_name][1],
                    parity,
                    samples[low_name],
                    samples[high_name],
                    args.budget,
                ),
            }

        evaluation = ids % 2 != parity
        per_candidate_scores = []
        for name in candidate_names:
            final_mse = ((combos[name] - targets) ** 2)[:, -1].mean(axis=1)
            per_candidate_scores.append(
                final_mse * np.maximum(SCORE_FLOOR, compute[name][1] / args.budget)
            )
        score_matrix = np.stack(per_candidate_scores, axis=1)
        oracle_scores = np.min(score_matrix[evaluation], axis=1)
        oracle_choice = np.argmin(score_matrix[evaluation], axis=1)
        parity_results[str(parity)] = {
            "lambdas": lambdas,
            "rules": rules,
            "best_rule_by_evaluation": min(
                rules, key=lambda name: rules[name]["adjusted_score"]["mean"]
            ),
            "oracle": {
                "adjusted_score": aggregate(oracle_scores),
                "choice_fractions": {
                    name: float(np.mean(oracle_choice == index))
                    for index, name in enumerate(candidate_names)
                },
            },
        }

    result = {
        "schema_version": "1.0",
        "experiment": "structural_qmc_adaptive_controller",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source": "structural_qmc_count_limit50",
            "rules": "fixed calibration-median thresholds; no target access at inference",
            "budget": args.budget,
        },
        "parities": parity_results,
    }
    output = RESULTS / "structural_qmc_controller_limit50.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def fit_binned_lambdas(
    delta: np.ndarray,
    baseline_error: np.ndarray,
    bins: np.ndarray,
    selected: np.ndarray,
    fallback: float,
) -> dict[int, float]:
    values = {}
    for bin_index in np.unique(bins):
        mask = selected[:, None] & (bins == bin_index)
        denominator = float(np.vdot(delta[mask], delta[mask]))
        value = (
            -float(np.vdot(delta[mask], baseline_error[mask])) / denominator
            if denominator
            else fallback
        )
        values[int(bin_index)] = min(1.0, max(0.0, value))
    return values


def apply_lambdas(k2: np.ndarray, delta: np.ndarray, bins: np.ndarray, lambdas: dict[int, float]) -> np.ndarray:
    scale = np.zeros_like(delta)
    for bin_index, value in lambdas.items():
        scale = np.where(bins == bin_index, value, scale)
    return k2 + scale * delta


def shrinkage_methods(
    ids: np.ndarray,
    k2: np.ndarray,
    prediction: np.ndarray,
    targets: np.ndarray,
    alpha2: np.ndarray,
    parity: int,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    selected = ids % 2 == parity
    delta = prediction[:, -1] - k2[:, -1]
    baseline_error = k2[:, -1] - targets[:, -1]
    global_lambda = fit_lambda(ids, k2, prediction, targets, parity)
    methods = {
        "global": (
            k2[:, -1] + global_lambda * delta,
            {"lambda": global_lambda},
        )
    }

    definitions: dict[str, tuple[np.ndarray, Any]] = {
        "abs_alpha_bins": (np.digitize(np.abs(alpha2), (1.0, 2.0, 3.0)), (1.0, 2.0, 3.0)),
        "signed_alpha_bins": (
            np.digitize(alpha2, (-3.0, -1.0, 1.0, 3.0)),
            (-3.0, -1.0, 1.0, 3.0),
        ),
    }
    k2_scale = np.sqrt(np.mean(k2[:, -1] ** 2, axis=1, keepdims=True))
    relative_correction = np.abs(delta) / np.maximum(k2_scale, 1e-12)
    quantiles = tuple(
        float(value) for value in np.quantile(relative_correction[selected], (0.25, 0.5, 0.75))
    )
    definitions["relative_correction_bins"] = (
        np.digitize(relative_correction, quantiles),
        quantiles,
    )

    for name, (bins, boundaries) in definitions.items():
        lambdas = fit_binned_lambdas(
            delta, baseline_error, bins, selected, global_lambda
        )
        methods[name] = (
            apply_lambdas(k2[:, -1], delta, bins, lambdas),
            {"boundaries": boundaries, "lambdas": lambdas},
        )

    mlp_feature = np.sqrt(np.mean(delta**2, axis=1)) / np.maximum(
        np.sqrt(np.mean(k2[:, -1] ** 2, axis=1)), 1e-12
    )
    threshold = float(np.median(mlp_feature[selected]))
    mlp_bins = np.broadcast_to((mlp_feature >= threshold)[:, None], delta.shape).astype(np.int64)
    lambdas = fit_binned_lambdas(delta, baseline_error, mlp_bins, selected, global_lambda)
    methods["mlp_disagreement_bins"] = (
        apply_lambdas(k2[:, -1], delta, mlp_bins, lambdas),
        {"threshold": threshold, "lambdas": lambdas},
    )
    return methods


def run_shrinkage(args: argparse.Namespace) -> Path:
    count_meta, count = load_stage("count")
    scramble_meta, scramble = load_stage("scramble")
    ids = count["mlp_ids"]
    targets = count["targets"]
    k2 = count["k2"]
    alpha2 = count["alpha2"]
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in count_meta["summary"]:
        flops_used, effective = compute_arrays(count_meta, name)
        sources[f"count__{name}"] = (count[f"prediction__{name}"], flops_used, effective)
    for name in scramble_meta["summary"]:
        flops_used, effective = compute_arrays(scramble_meta, name)
        sources[f"scramble__{name}"] = (
            scramble[f"prediction__{name}"],
            flops_used,
            effective,
        )

    parity_results = {}
    for parity in (0, 1):
        evaluation = ids % 2 != parity
        source_results = {}
        for source_name, (prediction, flops_used, effective) in sources.items():
            methods = shrinkage_methods(ids, k2, prediction, targets, alpha2, parity)
            method_results = {}
            for method_name, (final_prediction, parameters) in methods.items():
                final_per_mlp = np.mean(
                    (final_prediction[evaluation] - targets[evaluation, -1]) ** 2,
                    axis=1,
                )
                method_results[method_name] = {
                    "parameters": parameters,
                    "final_mse": aggregate(final_per_mlp),
                    "adjusted_score": aggregate(
                        final_per_mlp
                        * np.maximum(SCORE_FLOOR, effective[evaluation] / args.budget)
                    ),
                    "flop_only_adjusted_score": aggregate(
                        final_per_mlp
                        * np.maximum(SCORE_FLOOR, flops_used[evaluation] / args.budget)
                    ),
                }
            source_results[source_name] = method_results
        ranked = [
            (method_results[method]["adjusted_score"]["mean"], source, method)
            for source, method_results in source_results.items()
            for method in method_results
        ]
        ranked.sort()
        parity_results[str(parity)] = {
            "best": {"source": ranked[0][1], "method": ranked[0][2], "score": ranked[0][0]},
            "sources": source_results,
        }

    result = {
        "schema_version": "1.0",
        "experiment": "structural_qmc_conditional_shrinkage",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sources": list(sources),
            "budget": args.budget,
            "methods": [
                "global",
                "abs_alpha_bins",
                "signed_alpha_bins",
                "relative_correction_bins",
                "mlp_disagreement_bins",
            ],
        },
        "parities": parity_results,
    }
    output = RESULTS / "structural_qmc_shrinkage_limit50.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def global_crossfit_stage(stage: str, budget: int) -> dict[str, Any]:
    metadata, arrays = load_stage(stage)
    ids = arrays["mlp_ids"]
    k2 = arrays["k2"]
    targets = arrays["targets"]
    results = {}
    for name in metadata["summary"]:
        prediction = arrays[f"prediction__{name}"]
        flops_used, effective = compute_arrays(metadata, name)
        parity_rows = {}
        for parity in (0, 1):
            lambda_, combo = combo_prediction(ids, k2, prediction, targets, parity)
            evaluation = ids % 2 != parity
            final_per_mlp = ((combo[evaluation] - targets[evaluation]) ** 2)[:, -1].mean(
                axis=1
            )
            parity_rows[str(parity)] = {
                "lambda": lambda_,
                "raw_mse": float(np.mean(final_per_mlp)),
                "adjusted_score": float(
                    np.mean(
                        final_per_mlp
                        * np.maximum(SCORE_FLOOR, effective[evaluation] / budget)
                    )
                ),
                "flop_only_adjusted_score": float(
                    np.mean(
                        final_per_mlp
                        * np.maximum(SCORE_FLOOR, flops_used[evaluation] / budget)
                    )
                ),
            }
        results[name] = {
            "parities": parity_rows,
            "crossfit_raw_mse": float(
                np.mean([parity_rows[key]["raw_mse"] for key in ("0", "1")])
            ),
            "crossfit_adjusted_score": float(
                np.mean([parity_rows[key]["adjusted_score"] for key in ("0", "1")])
            ),
            "crossfit_flop_only_adjusted_score": float(
                np.mean(
                    [parity_rows[key]["flop_only_adjusted_score"] for key in ("0", "1")]
                )
            ),
        }
    return results


def run_final(args: argparse.Namespace) -> Path:
    stages = {
        stage: global_crossfit_stage(stage, args.budget)
        for stage in ("count", "ensemble", "scramble")
    }
    controller = json.loads(
        (RESULTS / "structural_qmc_controller_limit50.json").read_text(encoding="utf-8")
    )
    controller_names = set(controller["parities"]["0"]["rules"]) & set(
        controller["parities"]["1"]["rules"]
    )
    controller_crossfit = {
        name: float(
            np.mean(
                [
                    controller["parities"][parity]["rules"][name]["adjusted_score"]["mean"]
                    for parity in ("0", "1")
                ]
            )
        )
        for name in controller_names
    }

    shrinkage = json.loads(
        (RESULTS / "structural_qmc_shrinkage_limit50.json").read_text(encoding="utf-8")
    )
    shrinkage_crossfit = {}
    for source, methods in shrinkage["parities"]["0"]["sources"].items():
        for method in methods:
            key = f"{source}__{method}"
            shrinkage_crossfit[key] = float(
                np.mean(
                    [
                        shrinkage["parities"][parity]["sources"][source][method][
                            "adjusted_score"
                        ]["mean"]
                        for parity in ("0", "1")
                    ]
                )
            )

    global_rows = [
        {
            "stage": stage,
            "candidate": candidate,
            **metrics,
        }
        for stage, candidates in stages.items()
        for candidate, metrics in candidates.items()
    ]
    global_rows.sort(key=lambda row: row["crossfit_flop_only_adjusted_score"])
    result = {
        "schema_version": "1.0",
        "experiment": "structural_qmc_final_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ranking_basis": "mean of two cross-fitted parity directions",
        "global_candidates": global_rows,
        "best_global_by_flop_only": global_rows[0],
        "best_controller": {
            "name": min(controller_crossfit, key=controller_crossfit.get),
            "crossfit_adjusted_score": min(controller_crossfit.values()),
        },
        "best_shrinkage": {
            "name": min(shrinkage_crossfit, key=shrinkage_crossfit.get),
            "crossfit_adjusted_score": min(shrinkage_crossfit.values()),
        },
    }
    output = RESULTS / "structural_qmc_final_comparison.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controller", "shrinkage", "final"), required=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "controller":
        output = run_controller(args)
    elif args.mode == "shrinkage":
        output = run_shrinkage(args)
    else:
        output = run_final(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
