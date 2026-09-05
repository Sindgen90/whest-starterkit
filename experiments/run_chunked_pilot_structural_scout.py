"""Benchmark chunked structural tails and pilot-corrected neuron classes.

This is offline research code.  It intentionally uses NumPy for reporting and
classification diagnostics; submission-facing code remains grader-safe.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, SCORE_FLOOR, as_mlp
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_structural_tail_scout import (
    aggregate,
    antithetic_sobol_samples,
    classes,
    gaussian_classifier,
    sampled_prefix,
    sobol_base_integers,
    structural_tail,
)
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 32_768
FIXED_POINTSET_SEED = 0
SEED_OFFSET = 8_675_309
BASELINE_LAMBDA = 0.9768709009656374


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    threshold: float
    chunk_size: int
    pilot_samples: int


def mse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = (prediction - target) ** 2
    return float(error[-1].mean()), float(error.mean())


def initial_classes(
    alpha: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return classes(np.asarray(alpha, dtype=np.float64), threshold)


def demote_pilot_crossings(
    active: np.ndarray,
    dead: np.ndarray,
    uncertain: np.ndarray,
    pilot_preactivations: fnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Demote classified neurons whose pilot values cross the predicted sign."""
    active_keep = np.ones(active.size, dtype=bool)
    dead_keep = np.ones(dead.size, dtype=bool)
    if active.size:
        active_keep = np.asarray(
            fnp.all(pilot_preactivations[:, active] > 0.0, axis=0), dtype=bool
        )
    if dead.size:
        dead_keep = np.asarray(
            fnp.all(pilot_preactivations[:, dead] <= 0.0, axis=0), dtype=bool
        )

    demoted_active = active[~active_keep]
    demoted_dead = dead[~dead_keep]
    refined_uncertain = np.sort(
        np.concatenate((uncertain, demoted_active, demoted_dead))
    )
    return active[active_keep], dead[dead_keep], refined_uncertain, {
        "demoted_active": int(demoted_active.size),
        "demoted_dead": int(demoted_dead.size),
    }


def pilot_classes(
    mlp: Any,
    samples: fnp.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    threshold: float,
    pilot_samples: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, int],
    dict[str, Any] | None,
]:
    class1 = initial_classes(alpha1, threshold)
    class2 = initial_classes(alpha2, threshold)
    diagnostics = {
        "layer_1_demoted_active": 0,
        "layer_1_demoted_dead": 0,
        "layer_2_demoted_active": 0,
        "layer_2_demoted_dead": 0,
    }
    if not pilot_samples:
        return class1, class2, diagnostics, None

    half = samples.shape[0] // 2
    pilot_half = pilot_samples // 2
    pilot = fnp.concatenate(
        (samples[:pilot_half], samples[half : half + pilot_half]), axis=0
    )
    prefix_sums = []
    for w in mlp.weights[:-2]:
        pilot = fnp.maximum(pilot @ w, 0.0)
        prefix_sums.append(fnp.sum(pilot, axis=0))
    z1 = pilot @ mlp.weights[-2]
    h1 = fnp.maximum(z1, 0.0)
    z2 = h1 @ mlp.weights[-1]

    class1 = demote_pilot_crossings(*class1, z1)
    class2 = demote_pilot_crossings(*class2, z2)
    diagnostics.update(
        {
            "layer_1_demoted_active": class1[3]["demoted_active"],
            "layer_1_demoted_dead": class1[3]["demoted_dead"],
            "layer_2_demoted_active": class2[3]["demoted_active"],
            "layer_2_demoted_dead": class2[3]["demoted_dead"],
        }
    )
    cache = {
        "pilot_half": pilot_half,
        "prefix_sums": prefix_sums,
        "h1": h1,
        "z2": z2,
    }
    return class1[:3], class2[:3], diagnostics, cache


def class_counts(
    class1: tuple[np.ndarray, np.ndarray, np.ndarray],
    class2: tuple[np.ndarray, np.ndarray, np.ndarray],
    demotions: dict[str, int],
) -> dict[str, int]:
    active1, dead1, uncertain1 = class1
    active2, dead2, uncertain2 = class2
    return {
        "layer_1_active": int(active1.size),
        "layer_1_dead": int(dead1.size),
        "layer_1_uncertain": int(uncertain1.size),
        "layer_2_active": int(active2.size),
        "layer_2_dead": int(dead2.size),
        "layer_2_uncertain": int(uncertain2.size),
        **demotions,
    }


def encoded_alphas(
    width: int,
    classes_: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Encode already-refined classes for the existing structural-tail helper."""
    active, dead, _ = classes_
    alpha = np.zeros(width, dtype=np.float64)
    alpha[active] = 4.0
    alpha[dead] = -4.0
    return alpha


def refine_from_tail_pilot(
    mlp: Any,
    x: fnp.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    threshold: float,
    pilot_samples: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, int],
]:
    half = x.shape[0] // 2
    pilot_half = pilot_samples // 2
    pilot = fnp.concatenate((x[:pilot_half], x[half : half + pilot_half]), axis=0)
    z1 = pilot @ mlp.weights[-2]
    z2 = fnp.maximum(z1, 0.0) @ mlp.weights[-1]
    refined1 = demote_pilot_crossings(*initial_classes(alpha1, threshold), z1)
    refined2 = demote_pilot_crossings(*initial_classes(alpha2, threshold), z2)
    diagnostics = {
        "layer_1_demoted_active": refined1[3]["demoted_active"],
        "layer_1_demoted_dead": refined1[3]["demoted_dead"],
        "layer_2_demoted_active": refined2[3]["demoted_active"],
        "layer_2_demoted_dead": refined2[3]["demoted_dead"],
    }
    return refined1[:3], refined2[:3], diagnostics


def chunked_structural_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    class1: tuple[np.ndarray, np.ndarray, np.ndarray],
    class2: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk_size: int,
    pilot_cache: dict[str, Any] | None,
) -> fnp.ndarray:
    """Stream samples while retaining only sums and uncertain tail activations."""
    active1, _, uncertain1 = class1
    active2, _, uncertain2 = class2
    w1 = mlp.weights[-2]
    w2 = mlp.weights[-1]

    w1_uncertain = w1[:, uncertain1] if uncertain1.size else None
    collapsed = None
    if active1.size and uncertain2.size:
        collapsed = w1[:, active1] @ w2[active1, :][:, uncertain2]
    uncertain_to_uncertain = None
    if uncertain1.size and uncertain2.size:
        uncertain_to_uncertain = w2[uncertain1, :][:, uncertain2]

    if pilot_cache is None:
        prefix_sums = [fnp.zeros(mlp.width) for _ in mlp.weights[:-2]]
        uncertain1_sum = fnp.zeros(uncertain1.size)
        uncertain2_sum = fnp.zeros(uncertain2.size)
        ranges = [(0, samples.shape[0])]
    else:
        prefix_sums = list(pilot_cache["prefix_sums"])
        uncertain1_sum = (
            fnp.sum(pilot_cache["h1"][:, uncertain1], axis=0)
            if uncertain1.size
            else fnp.zeros(0)
        )
        uncertain2_sum = (
            fnp.sum(fnp.maximum(pilot_cache["z2"][:, uncertain2], 0.0), axis=0)
            if uncertain2.size
            else fnp.zeros(0)
        )
        pilot_half = pilot_cache["pilot_half"]
        half = samples.shape[0] // 2
        ranges = [(pilot_half, half), (half + pilot_half, samples.shape[0])]

    for range_start, range_stop in ranges:
        for start in range(range_start, range_stop, chunk_size):
            x = samples[start : min(start + chunk_size, range_stop)]
            for layer_index, w in enumerate(mlp.weights[:-2]):
                x = fnp.maximum(x @ w, 0.0)
                prefix_sums[layer_index] = fnp.add(
                    prefix_sums[layer_index], fnp.sum(x, axis=0)
                )

            h1_uncertain = None
            if uncertain1.size:
                h1_uncertain = fnp.maximum(x @ w1_uncertain, 0.0)
                uncertain1_sum = fnp.add(
                    uncertain1_sum, fnp.sum(h1_uncertain, axis=0)
                )

            if uncertain2.size:
                z2_uncertain = x @ collapsed if collapsed is not None else None
                if uncertain_to_uncertain is not None:
                    contribution = h1_uncertain @ uncertain_to_uncertain
                    z2_uncertain = (
                        contribution
                        if z2_uncertain is None
                        else fnp.add(z2_uncertain, contribution)
                    )
                if z2_uncertain is not None:
                    uncertain2_sum = fnp.add(
                        uncertain2_sum,
                        fnp.sum(fnp.maximum(z2_uncertain, 0.0), axis=0),
                    )

    scale = 1.0 / samples.shape[0]
    prefix_rows = [fnp.multiply(total, scale) for total in prefix_sums]
    mean1 = fnp.zeros(mlp.width, dtype=fnp.float32)
    if active1.size:
        mean1[active1] = prefix_rows[-1] @ w1[:, active1]
    if uncertain1.size:
        mean1[uncertain1] = fnp.multiply(uncertain1_sum, scale)

    mean2 = fnp.zeros(mlp.width, dtype=fnp.float32)
    if active2.size:
        mean2[active2] = mean1 @ w2[:, active2]
    if uncertain2.size:
        mean2[uncertain2] = fnp.multiply(uncertain2_sum, scale)
    return fnp.stack(prefix_rows + [mean1, mean2], axis=0)


def evaluate_variant(
    mlp: Any,
    samples: fnp.ndarray,
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, float | int]]:
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
        k2, alphas = gaussian_classifier(mlp)
        alpha1 = np.asarray(alphas[0], dtype=np.float64)
        alpha2 = np.asarray(alphas[1], dtype=np.float64)
        initial1 = initial_classes(alpha1, variant.threshold)
        initial2 = initial_classes(alpha2, variant.threshold)

        if variant.mode == "monolithic":
            x, prefix_rows = sampled_prefix(mlp, samples, samples, 0, "fnp")
            structural, _ = structural_tail(
                x,
                prefix_rows,
                mlp.weights[-2],
                mlp.weights[-1],
                alpha1,
                alpha2,
                variant.threshold,
            )
            refined1, refined2 = initial1, initial2
            demotions = {
                "layer_1_demoted_active": 0,
                "layer_1_demoted_dead": 0,
                "layer_2_demoted_active": 0,
                "layer_2_demoted_dead": 0,
            }
        elif variant.mode == "pilot_monolithic":
            x, prefix_rows = sampled_prefix(mlp, samples, samples, 0, "fnp")
            refined1, refined2, demotions = refine_from_tail_pilot(
                mlp,
                x,
                alpha1,
                alpha2,
                variant.threshold,
                variant.pilot_samples,
            )
            structural, _ = structural_tail(
                x,
                prefix_rows,
                mlp.weights[-2],
                mlp.weights[-1],
                encoded_alphas(mlp.width, refined1),
                encoded_alphas(mlp.width, refined2),
                3.0,
            )
        else:
            refined1, refined2, demotions, pilot_cache = pilot_classes(
                mlp,
                samples,
                alpha1,
                alpha2,
                variant.threshold,
                variant.pilot_samples,
            )
            structural = chunked_structural_prediction(
                mlp,
                samples,
                refined1,
                refined2,
                variant.chunk_size,
                pilot_cache,
            )

    runtime_s = time.perf_counter() - started
    residual_s = float(context.residual_wall_time_s or 0.0)
    flops_used = int(context.flops_used)
    counts = class_counts(refined1, refined2, demotions)
    counts.update(
        {
            "initial_layer_1_uncertain": int(initial1[2].size),
            "initial_layer_2_uncertain": int(initial2[2].size),
        }
    )
    return (
        np.asarray(k2, dtype=np.float64),
        np.asarray(structural, dtype=np.float64),
        counts,
        {
            "flops": flops_used,
            "residual_wall_time_s": residual_s,
            "runtime_s": runtime_s,
            "effective_compute": flops_used + LAMBDA_FLOPS_PER_SECOND * residual_s,
        },
    )


def fit_lambda(records: list[dict[str, Any]], name: str, parity: int) -> float:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        if record["mlp_id"] % 2 != parity:
            continue
        candidate = record["_variants"][name]
        k2 = candidate["k2"][-1]
        delta = candidate["structural"][-1] - k2
        target = record["_target"][-1]
        numerator += float(np.vdot(delta, k2 - target))
        denominator += float(np.vdot(delta, delta))
    value = -numerator / denominator if denominator else 0.0
    return min(1.0, max(0.0, value))


def summarize(
    records: list[dict[str, Any]], variants: list[Variant], args: argparse.Namespace
) -> dict[str, Any]:
    evaluation = [r for r in records if r["mlp_id"] % 2 != args.calibration_parity]
    result = {}
    for variant in variants:
        lambda_ = fit_lambda(records, variant.name, args.calibration_parity)
        direct_final = []
        direct_all = []
        combo_final = []
        combo_all = []
        adjusted = []
        flop_only_adjusted = []
        compute = []
        flops_used = []
        residual = []
        runtime = []
        exceeded = []
        counts = []
        reference_final = []
        reference_all = []
        reference_max_abs = []
        for record in evaluation:
            candidate = record["_variants"][variant.name]
            target = record["_target"]
            direct = candidate["structural"]
            combo = candidate["k2"] + lambda_ * (direct - candidate["k2"])
            final, all_layer = mse(direct, target)
            combo_mse, combo_all_mse = mse(combo, target)
            effective = candidate["compute"]["effective_compute"]
            multiplier = max(SCORE_FLOOR, effective / args.budget)
            direct_final.append(final)
            direct_all.append(all_layer)
            combo_final.append(combo_mse)
            combo_all.append(combo_all_mse)
            adjusted.append(combo_mse * multiplier)
            flop_only_adjusted.append(
                combo_mse * max(SCORE_FLOOR, candidate["compute"]["flops"] / args.budget)
            )
            compute.append(effective)
            flops_used.append(candidate["compute"]["flops"])
            residual.append(candidate["compute"]["residual_wall_time_s"])
            runtime.append(candidate["compute"]["runtime_s"])
            exceeded.append(effective > args.budget)
            counts.append(candidate["counts"])
            reference = record["_variants"].get("monolithic_t3")
            if reference is not None and variant.name != "monolithic_t3":
                difference = candidate["structural"] - reference["structural"]
                reference_final.append(float(np.mean(difference[-1] ** 2)))
                reference_all.append(float(np.mean(difference**2)))
                reference_max_abs.append(float(np.max(np.abs(difference))))
        result[variant.name] = {
            "configuration": {
                "mode": variant.mode,
                "threshold": variant.threshold,
                "chunk_size": variant.chunk_size,
                "pilot_samples": variant.pilot_samples,
            },
            "lambda": lambda_,
            "direct_final_mse": aggregate(direct_final),
            "direct_all_layer_mse": aggregate(direct_all),
            "combo_final_mse": aggregate(combo_final),
            "combo_all_layer_mse": aggregate(combo_all),
            "combo_adjusted_score": aggregate(adjusted),
            "combo_flop_only_adjusted_score": aggregate(flop_only_adjusted),
            "flops": aggregate(flops_used),
            "effective_compute": aggregate(compute),
            "residual_wall_time_s": aggregate(residual),
            "runtime_s": aggregate(runtime),
            "budget_exceeded_fraction": sum(exceeded) / len(exceeded),
            "vs_monolithic_t3_final_mse": aggregate(reference_final),
            "vs_monolithic_t3_all_layer_mse": aggregate(reference_all),
            "vs_monolithic_t3_max_abs": aggregate(reference_max_abs),
            "class_counts": {
                key: aggregate(row[key] for row in counts) for key in counts[0]
            },
        }
    return {"evaluation_mlps": len(evaluation), "variants": result}


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = []
    if not args.skip_baseline:
        variants.append(Variant("monolithic_t3", "monolithic", 3.0, N_SAMPLES, 0))
    if args.include_chunked_t3:
        variants.extend(
            Variant(f"chunk{size}_t3", "chunked", 3.0, size, 0)
            for size in args.chunk_sizes
        )
    for size in args.chunk_sizes:
        variants.extend(
            Variant(
                (
                    f"monolithic_t{args.pilot_threshold:g}_pilot{pilot}"
                    if args.pilot_mode == "monolithic"
                    else f"chunk{size}_t{args.pilot_threshold:g}_pilot{pilot}"
                ),
                "pilot_monolithic" if args.pilot_mode == "monolithic" else "chunked",
                args.pilot_threshold,
                size,
                pilot,
            )
            for pilot in args.pilot_samples
        )
    return list({variant.name: variant for variant in variants}.values())


def public_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        output.append(
            {
                "mlp_id": record["mlp_id"],
                "mlp_name": record["mlp_name"],
                "variants": {
                    name: {"counts": value["counts"], "compute": value["compute"]}
                    for name, value in record["_variants"].items()
                },
            }
        )
    return output


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    rows = rows[: args.limit_mlps]
    variants = build_variants(args)
    if not variants:
        raise ValueError("no variants selected")

    first_mlp = as_mlp(rows[0])
    setup_started = time.perf_counter()
    sobol_base = sobol_base_integers(
        N_SAMPLES, first_mlp.width, args.sobol_direction_file, "fnp"
    )
    samples = antithetic_sobol_samples(
        sobol_base, FIXED_POINTSET_SEED, SEED_OFFSET, "fnp"
    )
    setup_runtime_s = time.perf_counter() - setup_started

    records = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        candidates = {}
        mlp_id = int(row["mlp_id"])
        ordered_variants = variants
        if args.alternate_order and (mlp_id // 2) % 2:
            ordered_variants = list(reversed(variants))
        for variant in ordered_variants:
            k2, structural, counts, compute = evaluate_variant(mlp, samples, variant)
            candidates[variant.name] = {
                "k2": k2,
                "structural": structural,
                "counts": counts,
                "compute": compute,
            }
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "_target": np.asarray(row["all_layer_means"], dtype=np.float64),
                "_variants": candidates,
            }
        )
        print(f"{index}/{len(rows)} {row['mlp_name']} completed", flush=True)

    result = {
        "schema_version": "1.0",
        "experiment": "chunked_pilot_structural_tail_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "n_samples": N_SAMPLES,
            "budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "calibration_parity": args.calibration_parity,
            "alternate_order": args.alternate_order,
            "pilot_rule": "initial z-score classes; demote on any pilot sign crossing",
            "sobol_direction_file": str(args.sobol_direction_file),
            "fixed_pointset_seed": FIXED_POINTSET_SEED,
            "seed_offset": SEED_OFFSET,
            "variants": [variant.__dict__ for variant in variants],
        },
        "setup": {
            "runtime_s": setup_runtime_s,
            "sample_shape": list(samples.shape),
            "sample_bytes": int(samples.size * samples.itemsize),
        },
        "records": public_records(records),
        "summary": summarize(records, variants, args),
    }
    RESULTS.mkdir(exist_ok=True)
    variant_slug = "__".join(variant.name for variant in variants)
    order_slug = "balanced" if args.alternate_order else "fixed"
    output = RESULTS / (
        f"chunked_pilot_structural_limit{args.limit_mlps}_{order_slug}_{variant_slug}.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=6)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--include-chunked-t3", action="store_true")
    parser.add_argument("--pilot-samples", type=int, nargs="*", default=[])
    parser.add_argument("--pilot-threshold", type=float, default=2.5)
    parser.add_argument("--pilot-mode", choices=("monolithic", "chunked"), default="monolithic")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--fixed-order", dest="alternate_order", action="store_false")
    parser.set_defaults(alternate_order=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.chunk_sizes = sorted(set(args.chunk_sizes))
    args.pilot_samples = sorted(set(args.pilot_samples))
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    if any(size <= 0 or N_SAMPLES % size for size in args.chunk_sizes):
        raise SystemExit("chunk sizes must be positive divisors of 32768")
    if any(size <= 0 or size % 2 or size > N_SAMPLES for size in args.pilot_samples):
        raise SystemExit("pilot sample counts must be positive, even, and at most 32768")
    if not math.isfinite(args.pilot_threshold) or args.pilot_threshold <= 0.0:
        raise SystemExit("--pilot-threshold must be finite and positive")
    if not args.sobol_direction_file.exists():
        raise SystemExit(f"Sobol direction file not found: {args.sobol_direction_file}")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
