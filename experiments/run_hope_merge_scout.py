"""Evaluate HOPE-ranked activation merging and residual moment corrections."""
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
from run_deep_structural_tail_scout import K2_COEFFICIENT
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_hope_pruning_scout import summarize_candidate
from run_sparse_pilot_reinvestment_scout import P_THRESHOLDS
from run_structural_tail_scout import antithetic_sobol_samples, sobol_base_integers
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
BASE_SAMPLES = 32_768
SEED_OFFSET = 8_675_309
FIXED_POINTSET_SEED = 0
_COV_RESCALE_THRESHOLD = 1e100
EPS = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    n_samples: int = BASE_SAMPLES
    selection: str = "none"
    merges_per_layer: int = 0
    total_merges: int = 0
    correction: str = "none"
    correction_scale: float = 0.0
    extra_prune: int = 0


@dataclass(frozen=True)
class Merge:
    layer: int
    victim: int
    anchor: int
    coefficient: float
    residual_mean: float
    residual_variance: float
    residual_second_moment: float
    risk: float
    benefit: float


@dataclass(frozen=True)
class LayerPlan:
    dead: tuple[int, ...]
    merges: tuple[Merge, ...]
    extra_pruned: tuple[int, ...]
    alive: tuple[int, ...]


def candidates_for_args(args: argparse.Namespace) -> tuple[Candidate, ...]:
    if args.stage == "merge":
        return (
            Candidate("control_dead_m050"),
            Candidate("merge_fixed01", selection="fixed", merges_per_layer=1),
            Candidate("merge_fixed02", selection="fixed", merges_per_layer=2),
            Candidate("merge_fixed04", selection="fixed", merges_per_layer=4),
            Candidate("merge_fixed08", selection="fixed", merges_per_layer=8),
        )
    if args.stage == "global":
        total = args.winner_merges * (len(P_THRESHOLDS) - 2)
        return (
            Candidate("control_dead_m050"),
            Candidate(
                f"merge_fixed{args.winner_merges:02d}",
                selection="fixed",
                merges_per_layer=args.winner_merges,
            ),
            Candidate(
                f"merge_global{total:03d}",
                selection="global",
                total_merges=total,
            ),
        )
    selection = args.selection_mode
    selection_fields = {
        "selection": selection,
        "merges_per_layer": args.winner_merges if selection == "fixed" else 0,
        "total_merges": args.total_merges if selection == "global" else 0,
    }
    stem = (
        f"{selection}{args.winner_merges:02d}"
        if selection == "fixed"
        else f"global{args.total_merges:03d}"
    )
    if args.stage == "mean":
        return (
            Candidate("control_dead_m050"),
            Candidate(f"merge_{stem}", **selection_fields),
            Candidate(
                f"merge_{stem}_mean_s050",
                correction="mean",
                correction_scale=0.5,
                **selection_fields,
            ),
            Candidate(
                f"merge_{stem}_mean_s100",
                correction="mean",
                correction_scale=1.0,
                **selection_fields,
            ),
        )
    if args.stage == "variance":
        return (
            Candidate("control_dead_m050"),
            Candidate(f"merge_{stem}", **selection_fields),
            Candidate(
                f"merge_{stem}_mean_s{round(100 * args.mean_scale):03d}",
                correction="mean",
                correction_scale=args.mean_scale,
                **selection_fields,
            ),
            Candidate(
                f"merge_{stem}_meanvar_s050",
                correction="mean_variance",
                correction_scale=0.5,
                **selection_fields,
            ),
            Candidate(
                f"merge_{stem}_meanvar_s100",
                correction="mean_variance",
                correction_scale=1.0,
                **selection_fields,
            ),
        )
    if args.stage == "combine":
        correction_fields = {
            "correction": args.correction,
            "correction_scale": args.correction_scale,
        }
        return (
            Candidate("control_dead_m050"),
            Candidate(
                f"merge_{stem}_winner32768",
                **selection_fields,
                **correction_fields,
            ),
            Candidate(
                f"merge_{stem}_sobol33792",
                n_samples=33_792,
                **selection_fields,
                **correction_fields,
            ),
            Candidate(
                f"merge_{stem}_sobol34816",
                n_samples=34_816,
                **selection_fields,
                **correction_fields,
            ),
            Candidate(
                f"merge_{stem}_extra04",
                extra_prune=4,
                **selection_fields,
                **correction_fields,
            ),
        )
    raise ValueError(f"unknown stage {args.stage!r}")


def gaussian_merge_stats(
    mlp: Any,
    dtype: Any = fnp.float64,
    tail_depth: int = len(P_THRESHOLDS),
) -> tuple[
    fnp.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    int,
    float,
    int,
    float,
]:
    """Return K2 plus tail moments needed for analytic merge planning."""
    if tail_depth < 2 or tail_depth > mlp.depth:
        raise ValueError(f"invalid tail depth {tail_depth} for depth {mlp.depth}")
    start = mlp.depth - tail_depth
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
        mu = fnp.zeros(mlp.width, dtype=dtype)
        cov = fnp.eye(mlp.width, dtype=dtype)
        log_scale = 0.0
        rows: list[fnp.ndarray] = []
        alphas: list[fnp.ndarray] = []
        gains: list[fnp.ndarray] = []
        phis: list[fnp.ndarray] = []
        sigmas: list[fnp.ndarray] = []
        tail_means: list[fnp.ndarray] = []
        tail_covariances: list[fnp.ndarray] = []
        tail_log_scales: list[float] = []

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
            phis.append(phi)
            sigmas.append(sigma)
            if layer >= start:
                tail_means.append(raw_mean)
                tail_covariances.append(cov)
                tail_log_scales.append(log_scale)

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

        raw_covariances = [
            covariance * math.exp(2.0 * layer_log_scale)
            for covariance, layer_log_scale in zip(
                tail_covariances, tail_log_scales
            )
        ]
        tail_sigmas = [
            sigmas[layer] * math.exp(tail_log_scales[layer - start])
            for layer in range(start, mlp.depth)
        ]
        variance_sensitivity = [
            phis[layer] / (2.0 * fnp.maximum(tail_sigmas[layer - start], 1e-12))
            for layer in range(start, mlp.depth)
        ]
        vectors = np.asarray(
            fnp.stack(
                (
                    fnp.stack(tail_means, axis=0),
                    fnp.stack(gains[start:], axis=0),
                    fnp.stack(variance_sensitivity, axis=0),
                    fnp.stack(path_energy[start:], axis=0),
                ),
                axis=0,
            ),
            dtype=np.float64,
        )
        covariance_values = np.asarray(
            fnp.stack(raw_covariances, axis=0), dtype=np.float64
        )

    stats = {
        "means": vectors[0],
        "gains": vectors[1],
        "variance_sensitivity": vectors[2],
        "path_energy": vectors[3],
        "covariances": covariance_values,
    }
    return (
        k2,
        alpha_values,
        stats,
        int(k2_context.flops_used),
        float(k2_context.residual_wall_time_s or 0.0),
        int(merge_context.flops_used),
        float(merge_context.residual_wall_time_s or 0.0),
    )


def layer_pair_pool(
    layer: int,
    alpha: np.ndarray,
    threshold: float,
    mean: np.ndarray,
    covariance: np.ndarray,
    path_energy: np.ndarray,
    benefit: float,
    anchors_per_victim: int = 8,
) -> tuple[tuple[int, ...], list[Merge]]:
    dead = tuple(np.flatnonzero(alpha <= -threshold).tolist())
    survivors = np.flatnonzero(alpha > -threshold)
    if survivors.size < 2:
        return dead, []

    covariance = 0.5 * (covariance + covariance.T)
    gram = covariance + np.outer(mean, mean)
    diagonal = np.maximum(np.diag(gram), EPS)
    entries: list[Merge] = []
    survivor_set = survivors.tolist()
    for victim in survivor_set:
        anchors = np.asarray([index for index in survivor_set if index != victim])
        cross = gram[victim, anchors]
        residual_second = np.maximum(
            diagonal[victim] - cross * cross / diagonal[anchors], 0.0
        )
        take = min(anchors_per_victim, anchors.size)
        chosen = np.argpartition(residual_second, take - 1)[:take]
        for position in chosen:
            anchor = int(anchors[position])
            coefficient = float(gram[victim, anchor] / diagonal[anchor])
            residual_mean = float(mean[victim] - coefficient * mean[anchor])
            second = float(residual_second[position])
            residual_variance = max(0.0, second - residual_mean * residual_mean)
            entries.append(
                Merge(
                    layer=layer,
                    victim=victim,
                    anchor=anchor,
                    coefficient=coefficient,
                    residual_mean=residual_mean,
                    residual_variance=residual_variance,
                    residual_second_moment=second,
                    risk=second * max(float(path_energy[victim]), EPS),
                    benefit=benefit,
                )
            )
    return dead, entries


def select_disjoint(pool: list[Merge], count: int) -> list[Merge]:
    selected: list[Merge] = []
    used: set[int] = set()
    for merge in sorted(pool, key=lambda row: row.risk / max(row.benefit, EPS)):
        if merge.victim in used or merge.anchor in used:
            continue
        selected.append(merge)
        used.add(merge.victim)
        used.add(merge.anchor)
        if len(selected) >= count:
            break
    return selected


def build_plans(
    mlp: Any,
    alphas: np.ndarray,
    stats: dict[str, np.ndarray],
    candidate: Candidate,
) -> tuple[LayerPlan, ...]:
    start = mlp.depth - len(P_THRESHOLDS)
    middle_layers = list(range(start, mlp.depth - 2))
    pools: dict[int, list[Merge]] = {}
    dead_by_layer: dict[int, tuple[int, ...]] = {}
    previous_survivors = mlp.width
    for offset, layer in enumerate(middle_layers):
        survivor_count = int(np.sum(alphas[layer] > -P_THRESHOLDS[offset]))
        benefit = 2.0 * candidate.n_samples * (
            previous_survivors + survivor_count
        )
        dead, pool = layer_pair_pool(
            layer,
            alphas[layer],
            P_THRESHOLDS[offset],
            stats["means"][offset],
            stats["covariances"][offset],
            stats["path_energy"][offset],
            benefit,
        )
        dead_by_layer[layer] = dead
        pools[layer] = pool
        previous_survivors = survivor_count

    selected_by_layer: dict[int, list[Merge]] = {layer: [] for layer in middle_layers}
    if candidate.selection == "fixed":
        for layer in middle_layers:
            selected_by_layer[layer] = select_disjoint(
                pools[layer], candidate.merges_per_layer
            )
    elif candidate.selection == "global":
        combined = [merge for layer in middle_layers for merge in pools[layer]]
        used = {layer: set() for layer in middle_layers}
        for merge in sorted(
            combined, key=lambda row: row.risk / max(row.benefit, EPS)
        ):
            layer_used = used[merge.layer]
            if merge.victim in layer_used or merge.anchor in layer_used:
                continue
            selected_by_layer[merge.layer].append(merge)
            layer_used.add(merge.victim)
            layer_used.add(merge.anchor)
            if sum(len(rows) for rows in selected_by_layer.values()) >= candidate.total_merges:
                break

    plans = []
    for offset, layer in enumerate(middle_layers):
        dead = dead_by_layer[layer]
        merges = selected_by_layer[layer]
        victims = {merge.victim for merge in merges}
        anchors = {merge.anchor for merge in merges}
        survivors = [
            index
            for index in range(mlp.width)
            if index not in dead and index not in victims
        ]
        extra_count = min(
            candidate.extra_prune,
            max(0, len(survivors) - len(anchors) - 1),
        )
        extra_candidates = sorted(
            (index for index in survivors if index not in anchors),
            key=lambda index: alphas[layer, index],
        )
        extra = tuple(sorted(extra_candidates[:extra_count]))
        alive = tuple(index for index in survivors if index not in set(extra))
        plans.append(
            LayerPlan(
                dead=dead,
                merges=tuple(merges),
                extra_pruned=extra,
                alive=alive,
            )
        )
    return tuple(plans)


def classes(
    alpha: np.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int]]:
    active = np.flatnonzero(alpha >= threshold).tolist()
    dead = np.flatnonzero(alpha <= -threshold).tolist()
    uncertain = np.flatnonzero(np.abs(alpha) < threshold).tolist()
    return active, dead, uncertain


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    total = parts[0]
    for part in parts[1:]:
        total = fnp.add(total, part)
    return total


def folded_weight(
    weight: fnp.ndarray,
    previous_plan: LayerPlan | None,
    width: int,
) -> fnp.ndarray:
    if previous_plan is None:
        return weight
    if not previous_plan.merges:
        return weight[list(previous_plan.alive), :]
    positions = {neuron: position for position, neuron in enumerate(previous_plan.alive)}
    transform = np.zeros((len(previous_plan.alive), width), dtype=np.float32)
    transform[np.arange(len(previous_plan.alive)), previous_plan.alive] = 1.0
    for merge in previous_plan.merges:
        transform[positions[merge.anchor], merge.victim] += merge.coefficient
    return fnp.array(transform) @ weight


def correction_step(
    delta_mean: fnp.ndarray,
    delta_variance: fnp.ndarray,
    next_weight: fnp.ndarray,
    plan: LayerPlan,
    layer_mean: np.ndarray,
    layer_variance: np.ndarray,
    next_gain: np.ndarray,
    next_variance_sensitivity: np.ndarray,
    correction: str,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    pre_mean = delta_mean @ next_weight
    pre_variance = delta_variance @ (next_weight ** 2)
    residual_indices = []
    residual_means = []
    residual_variances = []
    for merge in plan.merges:
        residual_indices.append(merge.victim)
        residual_means.append(merge.residual_mean)
        residual_variances.append(merge.residual_variance)
    for neuron in plan.extra_pruned:
        residual_indices.append(neuron)
        residual_means.append(float(layer_mean[neuron]))
        residual_variances.append(float(max(layer_variance[neuron], 0.0)))

    if residual_indices:
        mean_injection = fnp.array(residual_means, dtype=fnp.float32)
        pre_mean = fnp.add(
            pre_mean, mean_injection @ next_weight[residual_indices, :]
        )
        if correction == "mean_variance":
            variance_injection = fnp.array(residual_variances, dtype=fnp.float32)
            pre_variance = fnp.add(
                pre_variance,
                variance_injection @ (next_weight[residual_indices, :] ** 2),
            )

    gain = fnp.array(next_gain.astype(np.float32))
    if correction == "mean_variance":
        sensitivity = fnp.array(next_variance_sensitivity.astype(np.float32))
        post_mean = gain * pre_mean + sensitivity * pre_variance
        post_variance = gain * gain * pre_variance
    else:
        post_mean = gain * pre_mean
        post_variance = fnp.zeros(delta_variance.shape, dtype=fnp.float32)
    return post_mean, post_variance


def structural_prediction(
    mlp: Any,
    samples: fnp.ndarray,
    k2: fnp.ndarray,
    alphas: np.ndarray,
    stats: dict[str, np.ndarray],
    plans: tuple[LayerPlan, ...],
    identity: fnp.ndarray,
    candidate: Candidate,
) -> tuple[fnp.ndarray, dict[str, Any]]:
    start = mlp.depth - len(P_THRESHOLDS)
    x = samples
    for weight in mlp.weights[:start]:
        x = fnp.maximum(x @ weight, 0.0)

    previous_plan = None
    delta_mean = fnp.zeros(mlp.width, dtype=fnp.float32)
    delta_variance = fnp.zeros(mlp.width, dtype=fnp.float32)
    diagnostics: dict[str, Any] = {}
    for offset, layer in enumerate(range(start, mlp.depth - 2)):
        plan = plans[offset]
        if not plan.alive:
            return k2, diagnostics
        effective_weight = folded_weight(mlp.weights[layer], previous_plan, mlp.width)
        x = fnp.maximum(x @ effective_weight[:, plan.alive], 0.0)

        if candidate.correction != "none":
            covariance_diagonal = np.diag(stats["covariances"][offset])
            delta_mean, delta_variance = correction_step(
                delta_mean,
                delta_variance,
                mlp.weights[layer + 1],
                plan,
                stats["means"][offset],
                covariance_diagonal,
                stats["gains"][offset + 1],
                stats["variance_sensitivity"][offset + 1],
                candidate.correction,
            )
        previous_plan = plan
        diagnostics[str(layer + 1)] = {
            "dead": len(plan.dead),
            "merges": len(plan.merges),
            "extra_pruned": len(plan.extra_pruned),
            "alive": len(plan.alive),
        }

    layer1 = mlp.depth - 2
    layer2 = mlp.depth - 1
    active1, dead1, uncertain1 = classes(alphas[layer1], P_THRESHOLDS[-2])
    active2, dead2, uncertain2 = classes(alphas[layer2], P_THRESHOLDS[-1])
    w1 = folded_weight(mlp.weights[layer1], previous_plan, mlp.width)
    w2 = mlp.weights[layer2]

    mean_x = fnp.mean(x, axis=0)
    mean1_parts: list[fnp.ndarray] = []
    h1_uncertain = None
    if active1:
        mean1_parts.append((mean_x @ w1[:, active1]) @ identity[active1, :])
    if uncertain1:
        h1_uncertain = fnp.maximum(x @ w1[:, uncertain1], 0.0)
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
            collapsed = w1[:, active1] @ w2[active1, :][:, uncertain2]
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

    if candidate.correction != "none":
        delta_mean = (delta_mean @ w2) * fnp.array(
            stats["gains"][-1].astype(np.float32)
        )
        if candidate.correction == "mean_variance":
            pre_variance = delta_variance @ (w2 ** 2)
            delta_mean = fnp.add(
                delta_mean,
                fnp.array(stats["variance_sensitivity"][-1].astype(np.float32))
                * pre_variance,
            )
        mean2 = fnp.add(mean2, delta_mean * candidate.correction_scale)

    prediction = fnp.concatenate((k2[:-1], mean2[None, :]), axis=0)
    diagnostics[str(layer1 + 1)] = {
        "dead": len(dead1), "active": len(active1), "uncertain": len(uncertain1)
    }
    diagnostics[str(layer2 + 1)] = {
        "dead": len(dead2), "active": len(active2), "uncertain": len(uncertain2)
    }
    return prediction, diagnostics


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
    candidates = candidates_for_args(args)
    width = as_mlp(rows[0]).width
    pointsets = build_pointsets(candidates, width, args.sobol_direction_file)
    identity = fnp.array(np.eye(width, dtype=np.float32))

    ids: list[int] = []
    targets: list[np.ndarray] = []
    k2_predictions: list[np.ndarray] = []
    predictions = {candidate.name: [] for candidate in candidates}
    compute = {candidate.name: [] for candidate in candidates}
    diagnostics = {candidate.name: [] for candidate in candidates}

    for row_number, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        (
            k2,
            alphas,
            stats,
            k2_flops,
            k2_residual,
            merge_flops,
            merge_residual,
        ) = gaussian_merge_stats(mlp)
        plans = {}
        plan_times = {}
        for candidate in candidates:
            started = time.perf_counter()
            plans[candidate.name] = build_plans(mlp, alphas, stats, candidate)
            plan_times[candidate.name] = time.perf_counter() - started

        ordered = candidates if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction, candidate_diagnostics = structural_prediction(
                    mlp,
                    pointsets[candidate.n_samples],
                    k2,
                    alphas,
                    stats,
                    plans[candidate.name],
                    identity,
                    candidate,
                )
            uses_merge = candidate.selection != "none"
            flops_used = k2_flops + int(context.flops_used)
            residual = k2_residual + float(context.residual_wall_time_s or 0.0)
            if uses_merge:
                flops_used += merge_flops
                residual += merge_residual + plan_times[candidate.name]
            runtime = time.perf_counter() - started
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            diagnostics[candidate.name].append(candidate_diagnostics)
            compute[candidate.name].append({
                "flops": float(flops_used),
                "effective_compute": float(
                    flops_used + LAMBDA_FLOPS_PER_SECOND * residual
                ),
                "residual_wall_time_s": residual,
                "runtime_s": runtime,
                "plan_time_s": plan_times[candidate.name],
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
        "experiment": "hope_merge_scout",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
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
    stem = f"hope_merge_{args.stage}_limit{args.limit_mlps}"
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
        "--stage", choices=("merge", "global", "mean", "variance", "combine"), required=True
    )
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--winner-merges", type=int, default=2)
    parser.add_argument("--selection-mode", choices=("fixed", "global"), default="fixed")
    parser.add_argument("--total-merges", type=int, default=28)
    parser.add_argument("--mean-scale", type=float, default=1.0)
    parser.add_argument("--correction", choices=("none", "mean", "mean_variance"), default="none")
    parser.add_argument("--correction-scale", type=float, default=1.0)
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
