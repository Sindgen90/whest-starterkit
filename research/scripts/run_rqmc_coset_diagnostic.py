"""Diagnose the six 4,096-point cosets embedded in the 24,576 RQMC lattice."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rqmc_sample_count_sweep import (  # noqa: E402
    BUDGET,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    select_rows,
)

from submissions.rqmc_lattice_002 import estimator as rqmc_source  # noqa: E402

N_FULL = 24_576
N_COSETS = 6
N_PER_COSET = 4_096
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
RQMC_SWEEP = (
    ROOT
    / "research"
    / "results"
    / "rqmc_sample_count_sweep_development_selection_4096_16384_24576_32768.json"
)
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"


def coset_means(activations: np.ndarray, n_cosets: int = N_COSETS) -> np.ndarray:
    """Return means of interlaced residue classes along the sample axis."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (samples, outputs)")
    if len(activations) % n_cosets:
        raise ValueError("sample count must be divisible by n_cosets")
    return np.stack(
        [activations[residue::n_cosets].mean(axis=0) for residue in range(n_cosets)],
        axis=0,
    )


def combinations_by_size(n_cosets: int = N_COSETS) -> dict[int, list[tuple[int, ...]]]:
    """Enumerate all one-, two-, three-, and full-coset combinations."""
    return {
        size: list(itertools.combinations(range(n_cosets), size))
        for size in (1, 2, 3, n_cosets)
    }


def combo_key(combo: tuple[int, ...]) -> str:
    return "+".join(str(index) for index in combo)


def equivalent_rqmc_flop_model() -> tuple[float, float]:
    """Infer the exact affine FLOP model from frozen ordinary-RQMC measurements."""
    result = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
    by_count = {int(row["n_samples"]): row for row in result["summary"]}
    low_count = 16_384
    high_count = 24_576
    low_flops = float(by_count[low_count]["flops"]["mean"])
    high_flops = float(by_count[high_count]["flops"]["mean"])
    slope = (high_flops - low_flops) / (high_count - low_count)
    intercept = high_flops - slope * high_count
    return slope, intercept


def equivalent_flops(n_samples: int) -> float:
    slope, intercept = equivalent_rqmc_flop_model()
    return intercept + slope * n_samples


def load_references() -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    rqmc_result = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
    direct_4096 = {
        int(row["mlp_id"]): row
        for row in rqmc_result["per_mlp"]
        if int(row["n_samples"]) == 4_096
    }
    k3_result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
    k3_rows = [
        row
        for row in k3_result["per_mlp"]
        if row["split"] == "development_selection" and row["method"] == "stack"
    ]
    k3_summary = {
        "final_mse": float(np.mean([row["final_mse"] for row in k3_rows])),
        "adjusted_score": float(np.mean([row["adjusted_score"] for row in k3_rows])),
        "flops": float(np.mean([row["flops"] for row in k3_rows])),
    }
    return direct_4096, k3_summary


def selection_summary(
    per_mlp: list[dict[str, Any]],
    combinations: dict[int, list[tuple[int, ...]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for size, combos in combinations.items():
        fixed_rows = []
        for combo in combos:
            key = combo_key(combo)
            values = [float(row["combination_mse"][str(size)][key]) for row in per_mlp]
            fixed_rows.append(
                {
                    "combo": list(combo),
                    "mse": aggregate(values),
                }
            )
        fixed_best = min(fixed_rows, key=lambda row: row["mse"]["mean"])
        oracle_values = [
            min(float(value) for value in row["combination_mse"][str(size)].values())
            for row in per_mlp
        ]
        oracle_choices = [
            min(
                row["combination_mse"][str(size)],
                key=row["combination_mse"][str(size)].get,
            )
            for row in per_mlp
        ]
        flops_used = equivalent_flops(size * N_PER_COSET)
        score_multiplier = max(SCORE_FLOOR, flops_used / BUDGET)
        result[str(size)] = {
            "n_samples": size * N_PER_COSET,
            "equivalent_flops": flops_used,
            "score_multiplier": score_multiplier,
            "fixed_best": {
                **fixed_best,
                "adjusted_score_mean": fixed_best["mse"]["mean"] * score_multiplier,
            },
            "oracle": {
                "mse": aggregate(oracle_values),
                "adjusted_score_mean": float(np.mean(oracle_values)) * score_multiplier,
                "choice_counts": {
                    key: oracle_choices.count(key) for key in sorted(set(oracle_choices))
                },
            },
            "all_fixed_combinations": fixed_rows,
        }
    return result


def error_correlation_summary(errors: np.ndarray) -> dict[str, Any]:
    """Summarize centered correlations and the exact cross-error decomposition."""
    flattened = errors.transpose(1, 0, 2).reshape(N_COSETS, -1)
    pearson = np.corrcoef(flattened)
    mse_per_coset = np.mean(flattened * flattened, axis=1)
    average_coset_mse = float(mse_per_coset.mean())
    cross_terms = []
    cosine = np.eye(N_COSETS, dtype=np.float64)
    for left, right in itertools.combinations(range(N_COSETS), 2):
        cross = float(np.mean(flattened[left] * flattened[right]))
        cross_terms.append(cross)
        denominator = math.sqrt(mse_per_coset[left] * mse_per_coset[right])
        cosine[left, right] = cosine[right, left] = cross / denominator
    average_cross = float(np.mean(cross_terms))
    effective_rho = average_cross / average_coset_mse
    formula_mse = average_coset_mse / N_COSETS * (
        1.0 + (N_COSETS - 1) * effective_rho
    )
    actual_average_error = flattened.mean(axis=0)
    return {
        "pearson_signed_error_matrix": pearson.tolist(),
        "uncentered_error_cosine_matrix": cosine.tolist(),
        "mean_off_diagonal_pearson": float(
            pearson[np.triu_indices(N_COSETS, k=1)].mean()
        ),
        "mean_off_diagonal_uncentered_cosine": float(
            cosine[np.triu_indices(N_COSETS, k=1)].mean()
        ),
        "average_individual_coset_mse": average_coset_mse,
        "average_cross_error": average_cross,
        "effective_rho_for_mse_identity": effective_rho,
        "mse_formula_prediction": formula_mse,
        "actual_six_coset_average_mse": float(
            np.mean(actual_average_error * actual_average_error)
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    selection = result["selection_summary"]
    direct = result["direct_rqmc_4096"]
    k3 = result["k3_reference"]
    correlation = result["error_correlation"]
    lines = [
        "# Six-coset diagnostic for the 24,576-point RQMC lattice",
        "",
        "The 24,576 pointwise activations were not stored by earlier experiments, so",
        "this diagnostic replayed each frozen development-selection MLP once. It then",
        "formed residue classes `k mod 6` without additional network evaluations.",
        "",
        "Fixed-best and oracle rows below are diagnostics selected on the same",
        "development data; they are not untouched-validation results or submission claims.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Individual cosets",
        "",
        "| residue | samples | mean final MSE | median | worst decile |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    one_rows = selection["1"]["all_fixed_combinations"]
    for row in sorted(one_rows, key=lambda item: item["combo"]):
        lines.append(
            f"| {row['combo'][0]} | 4096 | {row['mse']['mean']:.6g} | "
            f"{row['mse']['median']:.6g} | {row['mse']['worst_decile']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Compression diagnostic matrix",
            "",
            "| rule | samples | final MSE | adjusted diagnostic | equivalent FLOPs |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| current direct 4096 lattice | 4096 | {direct['final_mse']:.6g} | "
                f"{direct['adjusted_score']:.6g} | {direct['flops']:.6g} |"
            ),
        ]
    )
    for size in (1, 2, 3):
        row = selection[str(size)]
        fixed = row["fixed_best"]
        oracle = row["oracle"]
        lines.append(
            f"| best fixed {size}-coset `{combo_key(tuple(fixed['combo']))}` | "
            f"{row['n_samples']} | {fixed['mse']['mean']:.6g} | "
            f"{fixed['adjusted_score_mean']:.6g} | {row['equivalent_flops']:.6g} |"
        )
        lines.append(
            f"| oracle {size}-coset | {row['n_samples']} | {oracle['mse']['mean']:.6g} | "
            f"{oracle['adjusted_score_mean']:.6g} | {row['equivalent_flops']:.6g} |"
        )
    six = selection["6"]
    lines.extend(
        [
            (
                f"| all six cosets / original 24576 | 24576 | "
                f"{six['fixed_best']['mse']['mean']:.6g} | "
                f"{six['fixed_best']['adjusted_score_mean']:.6g} | "
                f"{six['equivalent_flops']:.6g} |"
            ),
            (
                f"| diagonal-K3 residual stack | -- | {k3['final_mse']:.6g} | "
                f"{k3['adjusted_score']:.6g} | {k3['flops']:.6g} |"
            ),
            "",
            "## Error dependence",
            "",
            "Pearson correlation of signed final-neuron errors, flattened across MLPs:",
            "",
            "| residue | 0 | 1 | 2 | 3 | 4 | 5 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for residue, correlations in enumerate(
        correlation["pearson_signed_error_matrix"]
    ):
        lines.append(
            f"| {residue} | " + " | ".join(f"{value:.3f}" for value in correlations) + " |"
        )
    lines.extend(
        [
            "",
            (
                "- Mean off-diagonal Pearson correlation of signed errors: "
                f"`{correlation['mean_off_diagonal_pearson']:.6f}`."
            ),
            (
                "- Mean off-diagonal uncentered error cosine: "
                f"`{correlation['mean_off_diagonal_uncentered_cosine']:.6f}`."
            ),
            (
                "- Effective rho in the six-way MSE identity: "
                f"`{correlation['effective_rho_for_mse_identity']:.6f}`."
            ),
            (
                "- MSE predicted by `V/6 * (1 + 5*rho)`: "
                f"`{correlation['mse_formula_prediction']:.6g}`."
            ),
            (
                "- Actual MSE of the six-coset average: "
                f"`{correlation['actual_six_coset_average_mse']:.6g}`."
            ),
            "",
            "## Decision",
            "",
            result["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not RQMC_SWEEP.exists():
        raise FileNotFoundError(RQMC_SWEEP)
    rows = select_rows("development_selection", args.limit)
    direct_by_id, k3_summary = load_references()
    combinations = combinations_by_size()
    rqmc_source.RQMC_SEED_OFFSET = RQMC_SEED_OFFSET
    rqmc_source.RQMC_MULTIPLIER = RQMC_MULTIPLIER
    per_mlp = []
    signed_errors = []
    replay_flops = []
    runtimes = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
            activations = rqmc_source.normal_lattice_samples(
                N_FULL,
                mlp.width,
                mlp.seed,
            )
            for weight in mlp.weights:
                activations = fnp.maximum(fnp.matmul(activations, weight), 0.0)
        runtime_s = time.perf_counter() - started
        final_activations = np.asarray(activations, dtype=np.float64)
        means = coset_means(final_activations)
        errors = means - target[None, :]
        signed_errors.append(errors)
        combination_mse: dict[str, dict[str, float]] = {}
        for size, combos in combinations.items():
            combination_mse[str(size)] = {
                combo_key(combo): float(
                    np.mean((means[np.asarray(combo)].mean(axis=0) - target) ** 2)
                )
                for combo in combos
            }
        mlp_id = int(row["mlp_id"])
        full_mse = combination_mse["6"][combo_key(tuple(range(N_COSETS)))]
        direct_4096_mse = float(direct_by_id[mlp_id]["final_mse"])
        per_mlp.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "combination_mse": combination_mse,
                "full_24576_mse": full_mse,
                "direct_4096_mse": direct_4096_mse,
                "replay_flops": int(context.flops_used),
                "runtime_s": runtime_s,
            }
        )
        replay_flops.append(float(context.flops_used))
        runtimes.append(runtime_s)
        print(
            f"{index}/{len(rows)} mlp_id={mlp_id} full_mse={full_mse:.6g} "
            f"best_coset={min(combination_mse['1'].values()):.6g}",
            flush=True,
        )

    selection = selection_summary(per_mlp, combinations)
    errors_array = np.stack(signed_errors, axis=0)
    correlation = error_correlation_summary(errors_array)
    direct_rows = [direct_by_id[int(row["mlp_id"])] for row in per_mlp]
    direct_summary = {
        "final_mse": float(np.mean([row["final_mse"] for row in direct_rows])),
        "adjusted_score": float(
            np.mean([row["adjusted_score"] for row in direct_rows])
        ),
        "flops": float(np.mean([row["flops"] for row in direct_rows])),
    }
    best_fixed_one = selection["1"]["fixed_best"]["mse"]["mean"]
    oracle_one = selection["1"]["oracle"]["mse"]["mean"]
    direct_mse = direct_summary["final_mse"]
    rho = correlation["effective_rho_for_mse_identity"]
    if best_fixed_one < direct_mse:
        decision = (
            "A fixed 24k coset beats the current direct 4k lattice on this development "
            "diagnostic, so Experiment 2 should compare that coset against generators "
            "constructed specifically for 4,096 points. "
        )
    else:
        decision = (
            "No fixed 24k coset beats the current direct 4k lattice, so blindly taking "
            "every sixth point is not justified. "
        )
    if oracle_one < 0.5 * direct_mse:
        decision += (
            "The oracle one-coset result is much stronger, indicating possible value in "
            "network-conditioned selection, but no cheap selector has been established."
        )
    elif rho < 0.2:
        decision += (
            "The low effective cross-error correlation shows that most of the 24k gain "
            "comes from averaging several differently wrong 4k cosets."
        )
    else:
        decision += (
            "The coset errors remain substantially correlated, limiting the benefit of "
            "point-set selection alone."
        )
    fixed_three = selection["3"]["fixed_best"]
    fixed_three_gap = (
        fixed_three["adjusted_score_mean"] / k3_summary["adjusted_score"] - 1.0
    )
    decision += (
        f" The in-sample best three-coset rule `{combo_key(tuple(fixed_three['combo']))}` "
        f"has adjusted diagnostic {fixed_three['adjusted_score_mean']:.6g}, "
        f"{100.0 * fixed_three_gap:+.2f}% versus K3, but it was selected on these same "
        "development MLPs and must be recalibrated on a disjoint split before evaluation."
    )

    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development_selection",
        "configuration": {
            "n_full": N_FULL,
            "n_cosets": N_COSETS,
            "n_per_coset": N_PER_COSET,
            "seed_offset": RQMC_SEED_OFFSET,
            "multiplier": RQMC_MULTIPLIER,
            "limit": args.limit,
            "holdout_touched": False,
            "validation_touched": False,
        },
        "source": {
            "rqmc_estimator": str(
                (ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py").relative_to(
                    ROOT
                )
            ),
            "rqmc_estimator_sha256": hashlib.sha256(
                (ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py").read_bytes()
            ).hexdigest(),
        },
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "replay": {
            "flops": aggregate(replay_flops),
            "runtime_s": aggregate(runtimes),
        },
        "direct_rqmc_4096": direct_summary,
        "k3_reference": k3_summary,
        "selection_summary": selection,
        "error_correlation": correlation,
        "decision": decision,
        "per_mlp": per_mlp,
    }
    output = RESULTS / "rqmc_24576_six_coset_diagnostic_development_selection.json"
    report = REPORTS / "rqmc_24576_six_coset_diagnostic_development_selection.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "direct_rqmc_4096": direct_summary,
                "selection_summary": selection,
                "error_correlation": correlation,
                "decision": decision,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
