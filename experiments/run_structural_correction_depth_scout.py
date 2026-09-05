"""Compare structural corrections, adaptive thresholds, and deeper tails."""
from __future__ import annotations

import argparse
import json
import math
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
from run_deep_structural_tail_scout import (
    K2_COEFFICIENT,
    SAMPLE_ASSET,
    dense_prediction,
    gaussian_classifier,
    summarize_candidate,
    summarize_signs,
)
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
F_THRESHOLDS = (3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0)
K_THRESHOLDS = (4.75, 4.5, 4.25, 4.0) + F_THRESHOLDS
M_THRESHOLDS = (5.75, 5.5, 5.25, 5.0) + K_THRESHOLDS


@dataclass(frozen=True)
class Candidate:
    name: str
    thresholds: tuple[float, ...]
    mode: str = "scalar"


CORRECTION_CANDIDATES = (
    Candidate("F_last8_control", F_THRESHOLDS),
    Candidate("G_last8_dead_tail", F_THRESHOLDS, "dead_tail"),
    Candidate("H_last8_amplification", F_THRESHOLDS, "amplification"),
    Candidate("I_last8_tail_risk", F_THRESHOLDS, "tail_risk"),
    Candidate("J_last10_tapered", (4.25, 4.0) + F_THRESHOLDS),
    Candidate("K_last12_tapered", K_THRESHOLDS),
)
DEEPER_CANDIDATES = (
    Candidate("F_last8_control", F_THRESHOLDS),
    Candidate("K_last12_control", K_THRESHOLDS),
    Candidate("L_last14_tapered", (5.25, 5.0) + K_THRESHOLDS),
    Candidate("M_last16_tapered", M_THRESHOLDS),
)
ASYMMETRIC_CANDIDATES = (
    Candidate("M_last16_symmetric_control", M_THRESHOLDS),
    Candidate("N_last16_dead_only_same", M_THRESHOLDS, "dead_only_all"),
    Candidate(
        "O_last16_dead_only_m025",
        tuple(value - 0.25 for value in M_THRESHOLDS[:-2]) + M_THRESHOLDS[-2:],
        "dead_only_all",
    ),
    Candidate(
        "P_last16_dead_only_m050",
        tuple(value - 0.5 for value in M_THRESHOLDS[:-2]) + M_THRESHOLDS[-2:],
        "dead_only_all",
    ),
    Candidate(
        "Q_last16_early_dead_m075",
        tuple(value - 0.75 for value in M_THRESHOLDS[:4]) + M_THRESHOLDS[4:],
        "dead_only_early",
    ),
)


def candidates_for_stage(stage: str) -> tuple[Candidate, ...]:
    if stage == "corrections":
        return CORRECTION_CANDIDATES
    if stage == "deeper":
        return DEEPER_CANDIDATES
    if stage == "asymmetric":
        return ASYMMETRIC_CANDIDATES
    raise ValueError(f"unknown stage {stage!r}")


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    scale = math.sqrt(2.0)
    return np.asarray(
        [0.5 * math.erfc(-float(value) / scale) for value in values],
        dtype=np.float64,
    )


def _scalar_classes(
    alpha: np.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int]]:
    active = np.flatnonzero(alpha >= threshold).tolist()
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    uncertain = np.flatnonzero(np.abs(alpha) < threshold).tolist()
    return active, dead, uncertain


def _threshold_array_classes(
    alpha: np.ndarray, thresholds: np.ndarray
) -> tuple[list[int], list[int], list[int]]:
    active_mask = alpha >= thresholds
    dead_mask = alpha <= -thresholds
    active = np.flatnonzero(active_mask).tolist()
    dead = np.flatnonzero(dead_mask).tolist()
    uncertain = np.flatnonzero(~(active_mask | dead_mask)).tolist()
    return active, dead, uncertain


def _downstream_amplification(
    mlp: Any, alphas: list[np.ndarray], start: int
) -> dict[int, np.ndarray]:
    """Approximate RMS final-layer Jacobian magnitude for each activation."""
    profiles = {mlp.depth - 1: np.ones(mlp.width, dtype=np.float64)}
    for layer in range(mlp.depth - 2, start - 1, -1):
        w_next = np.asarray(mlp.weights[layer + 1], dtype=np.float64)
        next_gate = _normal_cdf(alphas[layer + 1])
        downstream = next_gate * profiles[layer + 1] ** 2
        profiles[layer] = np.sqrt(np.maximum((w_next * w_next) @ downstream, 1e-24))
    return profiles


def _amplification_classes(
    mlp: Any,
    alphas: list[np.ndarray],
    candidate: Candidate,
    start: int,
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    profiles = _downstream_amplification(mlp, alphas, start)
    output = {}
    for offset, layer in enumerate(range(start, mlp.depth)):
        base = candidate.thresholds[offset]
        amplification = profiles[layer]
        normalizer = max(float(np.median(amplification)), 1e-12)
        relative = np.clip(amplification / normalizer, 0.1, 10.0)
        thresholds = np.clip(base + 0.35 * np.log(relative), base - 0.5, base + 0.75)
        output[layer] = _threshold_array_classes(alphas[layer], thresholds)
    return output


def _tail_risk_classes(
    mlp: Any,
    alphas: list[np.ndarray],
    candidate: Candidate,
    start: int,
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    """Reallocate the baseline classified count toward low-impact neurons."""
    output = {}
    for offset, layer in enumerate(range(start, mlp.depth)):
        alpha = alphas[layer]
        base = candidate.thresholds[offset]
        classified_count = int(np.count_nonzero(np.abs(alpha) >= base))
        if layer < mlp.depth - 1:
            outgoing = np.asarray(mlp.weights[layer + 1], dtype=np.float64)
            outgoing_norm = np.sqrt(np.sum(outgoing * outgoing, axis=1))
            outgoing_norm /= max(float(np.median(outgoing_norm)), 1e-12)
        else:
            outgoing_norm = np.ones(mlp.width, dtype=np.float64)
        tail_probability = _normal_cdf(-np.abs(alpha))
        risk = tail_probability * np.clip(outgoing_norm, 0.1, 10.0)
        eligible = np.flatnonzero(np.abs(alpha) >= max(2.5, base - 0.75))
        selected = eligible[np.argsort(risk[eligible])[:classified_count]]
        selected_mask = np.zeros(mlp.width, dtype=bool)
        selected_mask[selected] = True
        active = np.flatnonzero(selected_mask & (alpha >= 0.0)).tolist()
        dead = np.flatnonzero(selected_mask & (alpha < 0.0)).tolist()
        uncertain = np.flatnonzero(~selected_mask).tolist()
        output[layer] = (active, dead, uncertain)
    return output


def candidate_classes(
    mlp: Any, alphas: list[np.ndarray], candidate: Candidate
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    start = mlp.depth - len(candidate.thresholds)
    if candidate.mode == "amplification":
        return _amplification_classes(mlp, alphas, candidate, start)
    if candidate.mode == "tail_risk":
        return _tail_risk_classes(mlp, alphas, candidate, start)
    if candidate.mode in ("dead_only_all", "dead_only_early"):
        output = {}
        for offset, layer in enumerate(range(start, mlp.depth)):
            threshold = candidate.thresholds[offset]
            use_dead_only = layer < mlp.depth - 2 and (
                candidate.mode == "dead_only_all" or offset < 4
            )
            if use_dead_only:
                dead = np.flatnonzero(alphas[layer] <= -threshold).tolist()
                dead_mask = np.zeros(mlp.width, dtype=bool)
                dead_mask[dead] = True
                output[layer] = ([], dead, np.flatnonzero(~dead_mask).tolist())
            else:
                output[layer] = _scalar_classes(alphas[layer], threshold)
        return output
    return {
        layer: _scalar_classes(alphas[layer], candidate.thresholds[layer - start])
        for layer in range(start, mlp.depth)
    }


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
    identity: fnp.ndarray,
    classifications: dict[int, tuple[list[int], list[int], list[int]]],
    gaussian_means: fnp.ndarray,
    dead_tail_correction: bool,
) -> tuple[fnp.ndarray, dict[str, dict[str, int]]]:
    """Evaluate one structural candidate, optionally carrying omitted dead means."""
    start = min(classifications)
    x = samples
    rows = []
    counts = {}
    for w in mlp.weights[:start]:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))

    previous_alive = None
    previous_dead: list[int] = []
    previous_dead_values = None
    for layer in range(start, mlp.depth - 2):
        active, dead, uncertain = classifications[layer]
        alive = active + uncertain
        counts[str(layer + 1)] = {
            "active": len(active), "dead": len(dead), "uncertain": len(uncertain)
        }
        w = mlp.weights[layer]
        w_input = w if previous_alive is None else w[previous_alive, :]
        z_alive = x @ w_input[:, alive]
        if dead_tail_correction and previous_dead:
            constant = previous_dead_values @ w[previous_dead, :][:, alive]
            z_alive = fnp.add(z_alive, constant[None, :])
        parts = []
        if active:
            parts.append(z_alive[:, : len(active)])
        if uncertain:
            parts.append(fnp.maximum(z_alive[:, len(active) :], 0.0))
        x = parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=1)

        mean_parts = [fnp.mean(x, axis=0) @ identity[alive, :]]
        current_dead_values = None
        if dead_tail_correction and dead:
            current_dead_values = gaussian_means[layer][dead]
            mean_parts.append(current_dead_values @ identity[dead, :])
        rows.append(_sum_parts(mean_parts, mlp.width))
        previous_alive = alive
        previous_dead = dead if dead_tail_correction else []
        previous_dead_values = current_dead_values

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = classifications[layer1]
    active2, dead2, uncertain2 = classifications[layer2]
    for layer, groups in ((layer1, (active1, dead1, uncertain1)), (layer2, (active2, dead2, uncertain2))):
        counts[str(layer + 1)] = {
            "active": len(groups[0]), "dead": len(groups[1]), "uncertain": len(groups[2])
        }

    w1 = mlp.weights[layer1]
    w2 = mlp.weights[layer2]
    w1_input = w1 if previous_alive is None else w1[previous_alive, :]
    prebias1 = None
    if dead_tail_correction and previous_dead:
        prebias1 = previous_dead_values @ w1[previous_dead, :]

    mean_x = fnp.mean(x, axis=0)
    mean1_parts = []
    h1_uncertain = None
    if active1:
        active_mean = mean_x @ w1_input[:, active1]
        if prebias1 is not None:
            active_mean = fnp.add(active_mean, prebias1[active1])
        mean1_parts.append(active_mean @ identity[active1, :])
    if uncertain1:
        z1_uncertain = x @ w1_input[:, uncertain1]
        if prebias1 is not None:
            z1_uncertain = fnp.add(z1_uncertain, prebias1[uncertain1][None, :])
        h1_uncertain = fnp.maximum(z1_uncertain, 0.0)
        mean1_parts.append(fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :])
    dead1_values = None
    if dead_tail_correction and dead1:
        dead1_values = gaussian_means[layer1][dead1]
        mean1_parts.append(dead1_values @ identity[dead1, :])
    mean1 = _sum_parts(mean1_parts, mlp.width)

    mean2_parts = []
    if active2:
        mean2_parts.append((mean1 @ w2[:, active2]) @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            collapsed = w1_input[:, active1] @ w2[active1, :][:, uncertain2]
            z2_uncertain = x @ collapsed
            if prebias1 is not None:
                constant = prebias1[active1] @ w2[active1, :][:, uncertain2]
                z2_uncertain = fnp.add(z2_uncertain, constant[None, :])
        if uncertain1:
            contribution = h1_uncertain @ w2[uncertain1, :][:, uncertain2]
            z2_uncertain = contribution if z2_uncertain is None else fnp.add(z2_uncertain, contribution)
        if dead_tail_correction and dead1:
            constant = dead1_values @ w2[dead1, :][:, uncertain2]
            z2_uncertain = constant[None, :] if z2_uncertain is None else fnp.add(z2_uncertain, constant[None, :])
        if z2_uncertain is not None:
            mean2_parts.append(
                fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0) @ identity[uncertain2, :]
            )
    if dead_tail_correction and dead2:
        dead2_values = gaussian_means[layer2][dead2]
        mean2_parts.append(dead2_values @ identity[dead2, :])
    mean2 = _sum_parts(mean2_parts, mlp.width)
    return fnp.stack(rows + [mean1, mean2], axis=0), counts


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    payload = fnp.load(str(args.sample_asset))
    samples = payload["samples"]
    identity = payload["identity"]
    if samples.shape != (32_768, 256) or identity.shape != (256, 256):
        raise ValueError(f"unexpected sample asset shapes: {samples.shape}, {identity.shape}")
    candidates = candidates_for_stage(args.stage)

    ids = []
    targets = []
    k2_predictions = []
    dense_predictions = []
    predictions: dict[str, list[np.ndarray]] = {c.name: [] for c in candidates}
    compute: dict[str, list[dict[str, float | int]]] = {c.name: [] for c in candidates}
    counts: dict[str, list[dict[str, dict[str, int]]]] = {c.name: [] for c in candidates}
    sign_rows = []

    for row_index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        mlp_id = int(row["mlp_id"])
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
            k2, alphas = gaussian_classifier(mlp)
        gaussian_means = fnp.asarray(k2 / K2_COEFFICIENT)
        k2_residual = float(k2_context.residual_wall_time_s or 0.0)
        with flops.BudgetContext(flop_budget=10**15, quiet=True):
            dense, signs = dense_prediction(mlp, samples, alphas, candidates)
        dense_predictions.append(np.asarray(dense, dtype=np.float64))
        sign_rows.append(signs)

        ordered = candidates if (mlp_id // 2) % 2 == 0 else tuple(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                classifications = candidate_classes(mlp, alphas, candidate)
                prediction, candidate_counts = structural_prediction(
                    mlp,
                    samples,
                    identity,
                    classifications,
                    gaussian_means,
                    candidate.mode == "dead_tail",
                )
            runtime_s = time.perf_counter() - started
            residual_s = k2_residual + float(context.residual_wall_time_s or 0.0)
            flops_used = int(k2_context.flops_used) + int(context.flops_used)
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            counts[candidate.name].append(candidate_counts)
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
    dense_array = np.stack(dense_predictions)
    prediction_arrays = {name: np.stack(value) for name, value in predictions.items()}
    parity_summaries = {}
    for parity in (0, 1):
        parity_summaries[str(parity)] = {
            candidate.name: summarize_candidate(
                candidate,
                ids_array,
                k2_array,
                prediction_arrays[candidate.name],
                dense_array,
                targets_array,
                compute[candidate.name],
                counts[candidate.name],
                parity,
                args.budget,
            )
            for candidate in candidates
        }
    fields = (
        "direct_final_mse", "combo_final_mse", "combo_all_layer_mse",
        "combo_adjusted_score", "combo_flop_only_adjusted_score",
        "structural_vs_dense_final_mse", "flops", "effective_compute",
    )
    crossfit = {}
    for candidate in candidates:
        p0 = parity_summaries["0"][candidate.name]
        p1 = parity_summaries["1"][candidate.name]
        crossfit[candidate.name] = {
            field: 0.5 * (p0[field]["mean"] + p1[field]["mean"])
            for field in fields
        }
        crossfit[candidate.name]["lambda_mean"] = 0.5 * (p0["lambda"] + p1["lambda"])

    result = {
        "schema_version": "1.0",
        "experiment": "structural_correction_depth_scout",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET, "revision": REVISION, "split": "mini",
            "limit_mlps": args.limit_mlps, "budget": args.budget,
            "sample_asset": str(args.sample_asset), "balanced_order": True,
            "dense_is_diagnostic_only": True,
            "candidates": [asdict(candidate) for candidate in candidates],
        },
        "summary": {
            "parity_crossfit": parity_summaries,
            "crossfit_mean": crossfit,
            "rankings": {
                field: sorted(crossfit, key=lambda name: crossfit[name][field])
                for field in (
                    "combo_final_mse", "combo_adjusted_score",
                    "combo_flop_only_adjusted_score", "structural_vs_dense_final_mse",
                )
            },
            "wrong_sign_diagnostics_all_mlps": summarize_signs(sign_rows),
        },
        "per_mlp": [
            {
                "mlp_id": int(mlp_id),
                "compute": {c.name: compute[c.name][index] for c in candidates},
                "class_counts": {c.name: counts[c.name][index] for c in candidates},
            }
            for index, mlp_id in enumerate(ids)
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    stem = (
        "structural_correction_depth"
        if args.stage == "corrections"
        else "structural_deeper_tail"
        if args.stage == "deeper"
        else "structural_asymmetric_dead"
    )
    json_path = RESULTS / f"{stem}_limit{args.limit_mlps}.json"
    cache_path = RESULTS / f"{stem}_limit{args.limit_mlps}.npz"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        cache_path,
        mlp_ids=ids_array,
        targets=targets_array,
        k2=k2_array,
        dense=dense_array,
        **{f"prediction__{name}": value for name, value in prediction_arrays.items()},
    )
    return json_path, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("corrections", "deeper", "asymmetric"),
        default="corrections",
    )
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--sample-asset", type=Path, default=SAMPLE_ASSET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    if not args.sample_asset.exists():
        raise SystemExit(f"sample asset not found: {args.sample_asset}")
    json_path, cache_path = run(args)
    print(f"wrote {json_path.relative_to(ROOT)} and {cache_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
