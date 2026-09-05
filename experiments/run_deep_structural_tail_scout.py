"""Compare structural sampling in the last 2, 3, 4, and 6 MLP layers.

Dense 32k propagation is evaluated only as a diagnostic reference. Candidate
compute always includes an independent sampled propagation plus corrected K2.
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
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ASSET = (
    ROOT / "submissions" / "structural_tail_sobol_32768" / "samples.npz"
)
K2_COEFFICIENT = 0.9918858550287327
_COV_RESCALE_THRESHOLD = 1e100


@dataclass(frozen=True)
class Candidate:
    name: str
    thresholds: tuple[float, ...]


BASE_CANDIDATES = (
    Candidate("current_last2_t3", (3.0, 3.0)),
    Candidate("A_last3_t3", (3.0, 3.0, 3.0)),
    Candidate("B_last4_conservative", (3.5, 3.5, 3.0, 3.0)),
    Candidate("C_last6_tapered", (3.5, 3.4, 3.3, 3.2, 3.0, 3.0)),
)
LAST8_CANDIDATES = (
    Candidate("current_last2_t3", (3.0, 3.0)),
    Candidate("C_last6_tapered", (3.5, 3.4, 3.3, 3.2, 3.0, 3.0)),
    Candidate(
        "D_last8_conservative",
        (4.25, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 3.0),
    ),
    Candidate(
        "E_last8_balanced",
        (4.0, 3.8, 3.6, 3.5, 3.4, 3.2, 3.0, 3.0),
    ),
    Candidate(
        "F_last8_aggressive",
        (3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0),
    ),
)


def candidates_for_stage(stage: str) -> tuple[Candidate, ...]:
    if stage == "base":
        return BASE_CANDIDATES
    if stage == "last8":
        return LAST8_CANDIDATES
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


def gaussian_classifier(mlp: Any) -> tuple[fnp.ndarray, list[np.ndarray]]:
    """Return corrected-K2 means and Gaussian z-scores for every layer."""
    mu = fnp.zeros(mlp.width)
    cov = fnp.eye(mlp.width)
    log_scale = 0.0
    rows = []
    alphas = []
    for w in mlp.weights:
        cov_diag = fnp.diag(cov)
        max_var = float(fnp.max(cov_diag))
        if max_var > _COV_RESCALE_THRESHOLD:
            scale = float(fnp.sqrt(max_var))
            mu = mu / scale
            cov = cov / (scale * scale)
            log_scale += float(fnp.log(scale))

        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        alphas.append(np.asarray(alpha, dtype=np.float64))

        phi_alpha = flops.stats.norm.pdf(alpha)
        Phi_alpha = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        ez2 = (
            (mu_pre * mu_pre + var_pre) * Phi_alpha
            + mu_pre * sigma_pre * phi_alpha
        )
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)
        sigma64 = fnp.asarray(sigma_pre, dtype=fnp.float64)
        gain64 = fnp.where(
            sigma64 > 1e-12,
            fnp.asarray(Phi_alpha, dtype=fnp.float64),
            0.0,
        )
        gain = fnp.array(gain64.astype(fnp.float32))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        rows.append(mu * float(fnp.exp(log_scale)))

    return fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT), alphas


def classes(alpha: np.ndarray, threshold: float) -> tuple[list[int], list[int], list[int]]:
    active = np.flatnonzero(alpha >= threshold).tolist()
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    uncertain = np.flatnonzero(np.abs(alpha) < threshold).tolist()
    return active, dead, uncertain


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    result = parts[0]
    for part in parts[1:]:
        result = fnp.add(result, part)
    return result


def deep_structural_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    alphas: list[np.ndarray],
    identity: fnp.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[fnp.ndarray, dict[str, dict[str, int]]]:
    """Propagate samples with structural pruning, collapsing the final two layers."""
    structural_depth = len(thresholds)
    if structural_depth < 2 or structural_depth > mlp.depth:
        raise ValueError("structural depth must be between 2 and MLP depth")

    x = samples
    rows = []
    start = mlp.depth - structural_depth
    for w in mlp.weights[:start]:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))

    previous_alive: list[int] | None = None
    counts: dict[str, dict[str, int]] = {}
    # Earlier structural layers retain samples for active and uncertain neurons.
    for layer_index in range(start, mlp.depth - 2):
        threshold = thresholds[layer_index - start]
        active, dead, uncertain = classes(alphas[layer_index], threshold)
        alive = active + uncertain
        counts[str(layer_index + 1)] = {
            "active": len(active), "dead": len(dead), "uncertain": len(uncertain)
        }
        if not alive:
            raise RuntimeError(f"all neurons classified dead at layer {layer_index + 1}")
        w = mlp.weights[layer_index]
        w_input = w if previous_alive is None else w[previous_alive, :]
        z_alive = x @ w_input[:, alive]
        parts = []
        if active:
            parts.append(z_alive[:, : len(active)])
        if uncertain:
            parts.append(fnp.maximum(z_alive[:, len(active) :], 0.0))
        x = parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=1)
        mean_alive = fnp.mean(x, axis=0)
        rows.append(mean_alive @ identity[alive, :])
        previous_alive = alive

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = classes(alphas[layer1], thresholds[-2])
    active2, dead2, uncertain2 = classes(alphas[layer2], thresholds[-1])
    counts[str(layer1 + 1)] = {
        "active": len(active1), "dead": len(dead1), "uncertain": len(uncertain1)
    }
    counts[str(layer2 + 1)] = {
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
        mean1_parts.append(fnp.mean(h1_uncertain, axis=0) @ identity[uncertain1, :])
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
    return fnp.stack(rows + [mean1, mean2], axis=0), counts


def diagnostic_requests(
    depth: int, candidates: tuple[Candidate, ...]
) -> dict[int, set[float]]:
    requests: dict[int, set[float]] = {}
    for candidate in candidates:
        start = depth - len(candidate.thresholds)
        for offset, threshold in enumerate(candidate.thresholds):
            requests.setdefault(start + offset, set()).add(threshold)
    return requests


def sign_diagnostics(
    z: fnp.ndarray, alpha: np.ndarray, threshold: float
) -> dict[str, float | None]:
    active, dead, _ = classes(alpha, threshold)
    result: dict[str, float | None] = {
        "active_wrong_sign_fraction": None,
        "active_worst_neuron_wrong_sign_fraction": None,
        "dead_wrong_sign_fraction": None,
        "dead_worst_neuron_wrong_sign_fraction": None,
    }
    if active:
        rates = np.asarray(fnp.mean(z[:, active] <= 0.0, axis=0), dtype=np.float64)
        result["active_wrong_sign_fraction"] = float(rates.mean())
        result["active_worst_neuron_wrong_sign_fraction"] = float(rates.max())
    if dead:
        rates = np.asarray(fnp.mean(z[:, dead] > 0.0, axis=0), dtype=np.float64)
        result["dead_wrong_sign_fraction"] = float(rates.mean())
        result["dead_worst_neuron_wrong_sign_fraction"] = float(rates.max())
    return result


def dense_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    alphas: list[np.ndarray],
    candidates: tuple[Candidate, ...],
) -> tuple[fnp.ndarray, dict[str, dict[str, float | None]]]:
    requests = diagnostic_requests(mlp.depth, candidates)
    x = samples
    rows = []
    diagnostics = {}
    for layer_index, w in enumerate(mlp.weights):
        z = x @ w
        for threshold in sorted(requests.get(layer_index, ())):
            key = f"layer_{layer_index + 1}_threshold_{threshold:g}"
            diagnostics[key] = sign_diagnostics(z, alphas[layer_index], threshold)
        x = fnp.maximum(z, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0), diagnostics


def fit_lambda(
    ids: np.ndarray,
    k2: np.ndarray,
    sampled: np.ndarray,
    targets: np.ndarray,
    calibration_parity: int,
) -> float:
    selected = ids % 2 == calibration_parity
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
    dense: np.ndarray,
    targets: np.ndarray,
    compute: list[dict[str, float | int]],
    counts: list[dict[str, dict[str, int]]],
    calibration_parity: int,
    budget: int,
) -> dict[str, Any]:
    evaluation = ids % 2 != calibration_parity
    lambda_ = fit_lambda(ids, k2, sampled, targets, calibration_parity)
    direct = sampled[evaluation]
    combo = k2[evaluation] + lambda_ * (direct - k2[evaluation])
    target = targets[evaluation]
    combo_error = (combo - target) ** 2
    direct_error = (direct - target) ** 2
    structural_error = (direct - dense[evaluation]) ** 2
    eval_compute = [row for row, keep in zip(compute, evaluation) if keep]
    flops_used = np.asarray([row["flops"] for row in eval_compute], dtype=np.float64)
    effective = np.asarray(
        [row["effective_compute"] for row in eval_compute], dtype=np.float64
    )
    final_per_mlp = combo_error[:, -1].mean(axis=1)
    selected_counts = [row for row, keep in zip(counts, evaluation) if keep]
    count_summary = {}
    for layer in sorted(selected_counts[0], key=int):
        count_summary[layer] = {
            field: aggregate(row[layer][field] for row in selected_counts)
            for field in ("active", "dead", "uncertain")
        }
    return {
        "configuration": asdict(candidate),
        "lambda": lambda_,
        "evaluation_mlps": int(evaluation.sum()),
        "direct_final_mse": aggregate(direct_error[:, -1].mean(axis=1)),
        "combo_final_mse": aggregate(final_per_mlp),
        "combo_all_layer_mse": aggregate(combo_error.mean(axis=(1, 2))),
        "combo_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, effective / budget)
        ),
        "combo_flop_only_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, flops_used / budget)
        ),
        "structural_vs_dense_final_mse": aggregate(
            structural_error[:, -1].mean(axis=1)
        ),
        "structural_vs_dense_all_layer_mse": aggregate(
            structural_error.mean(axis=(1, 2))
        ),
        "structural_vs_dense_mse_by_layer": [
            float(structural_error[:, layer].mean())
            for layer in range(structural_error.shape[1])
        ],
        "flops": aggregate(flops_used),
        "effective_compute": aggregate(effective),
        "residual_wall_time_s": aggregate(
            row["residual_wall_time_s"] for row in eval_compute
        ),
        "runtime_s": aggregate(row["runtime_s"] for row in eval_compute),
        "budget_exceeded_fraction": float(np.mean(effective > budget)),
        "class_counts": count_summary,
    }


def summarize_signs(
    rows: list[dict[str, dict[str, float | None]]]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    keys = sorted({key for row in rows for key in row})
    output = {}
    for key in keys:
        fields = rows[0].get(key, {}) or next(row[key] for row in rows if key in row)
        output[key] = {
            field: aggregate(
                row[key][field]
                for row in rows
                if key in row and row[key][field] is not None
            )
            for field in fields
        }
    return output


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
        k2_residual = float(k2_context.residual_wall_time_s or 0.0)

        with flops.BudgetContext(flop_budget=10**15, quiet=True):
            dense, signs = dense_prediction(mlp, samples, alphas, candidates)
        dense_predictions.append(np.asarray(dense, dtype=np.float64))
        sign_rows.append(signs)

        ordered = candidates if (mlp_id // 2) % 2 == 0 else tuple(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_counts = deep_structural_prediction(
                    mlp, samples, alphas, identity, candidate.thresholds
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

    crossfit = {}
    for candidate in candidates:
        p0 = parity_summaries["0"][candidate.name]
        p1 = parity_summaries["1"][candidate.name]
        fields = (
            "direct_final_mse",
            "combo_final_mse",
            "combo_all_layer_mse",
            "combo_adjusted_score",
            "combo_flop_only_adjusted_score",
            "structural_vs_dense_final_mse",
            "flops",
            "effective_compute",
        )
        crossfit[candidate.name] = {
            field: 0.5 * (p0[field]["mean"] + p1[field]["mean"])
            for field in fields
        }
        crossfit[candidate.name]["lambda_mean"] = 0.5 * (p0["lambda"] + p1["lambda"])

    rankings = {
        field: sorted(crossfit, key=lambda name: crossfit[name][field])
        for field in (
            "combo_final_mse", "combo_adjusted_score",
            "combo_flop_only_adjusted_score", "structural_vs_dense_final_mse"
        )
    }
    result = {
        "schema_version": "1.0",
        "experiment": "deep_structural_tail_scout",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "budget": args.budget,
            "sample_asset": str(args.sample_asset),
            "balanced_order": True,
            "dense_is_diagnostic_only": True,
            "candidates": [asdict(candidate) for candidate in candidates],
        },
        "summary": {
            "parity_crossfit": parity_summaries,
            "crossfit_mean": crossfit,
            "rankings": rankings,
            "wrong_sign_diagnostics_all_mlps": summarize_signs(sign_rows),
        },
        "per_mlp": [
            {
                "mlp_id": int(mlp_id),
                "compute": {candidate.name: compute[candidate.name][index] for candidate in candidates},
                "class_counts": {candidate.name: counts[candidate.name][index] for candidate in candidates},
                "wrong_sign_diagnostics": sign_rows[index],
            }
            for index, mlp_id in enumerate(ids)
        ],
    }

    RESULTS.mkdir(exist_ok=True)
    stem = "deep_structural_tail" if args.stage == "base" else f"deep_structural_tail_{args.stage}"
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
    parser.add_argument("--stage", choices=("base", "last8"), default="base")
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
