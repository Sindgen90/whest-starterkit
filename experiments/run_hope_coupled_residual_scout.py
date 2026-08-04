"""Test coupled full-minus-HOPE residual corrections at several sample sizes."""
from __future__ import annotations

import argparse
import json
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
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE
from run_hope_merge_optimized_scout import (
    OPTIMIZED,
    gaussian_top_pair_payload,
    optimized_plans,
)
from run_hope_merge_scout import Candidate, build_pointsets, structural_prediction
from run_hope_pruning_scout import summarize_candidate
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
MAIN_SAMPLES = 32_768
CORRECTION_SAMPLES = (2_048, 4_096, 8_192)


@dataclass(frozen=True)
class ResidualCandidate:
    name: str
    n_samples: int
    correction_samples: int


CANDIDATES = tuple(
    ResidualCandidate(
        name=f"hope_residual_{correction_samples:05d}",
        n_samples=MAIN_SAMPLES,
        correction_samples=correction_samples,
    )
    for correction_samples in CORRECTION_SAMPLES
)


def full_sample_final(mlp: Any, samples: fnp.ndarray) -> fnp.ndarray:
    x = samples
    for weight in mlp.weights:
        x = fnp.maximum(x @ weight, 0.0)
    return fnp.mean(x, axis=0)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    width = as_mlp(rows[0]).width
    pointset_candidates = (Candidate("base", n_samples=MAIN_SAMPLES),) + tuple(
        Candidate(f"correction_{count}", n_samples=count)
        for count in CORRECTION_SAMPLES
    )
    pointsets = build_pointsets(
        pointset_candidates, width, args.sobol_direction_file
    )
    identity = fnp.array(np.eye(width, dtype=np.float32))

    ids, targets, k2_predictions = [], [], []
    predictions = {candidate.name: [] for candidate in CANDIDATES}
    compute = {candidate.name: [] for candidate in CANDIDATES}
    diagnostics = {candidate.name: [] for candidate in CANDIDATES}

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
        ) = gaussian_top_pair_payload(mlp, dtype=fnp.float32)
        plan_started = time.perf_counter()
        plans = optimized_plans(mlp, alphas, payload)
        plan_time = time.perf_counter() - plan_started

        with flops.BudgetContext(flop_budget=10**15, quiet=True) as base_context:
            reduced_main, base_diagnostics = structural_prediction(
                mlp,
                pointsets[MAIN_SAMPLES],
                k2,
                alphas,
                {},
                plans,
                identity,
                OPTIMIZED,
            )
        base_flops = int(base_context.flops_used)
        base_residual = float(base_context.residual_wall_time_s or 0.0)

        ordered = CANDIDATES if (int(row["mlp_id"]) // 2) % 2 == 0 else tuple(reversed(CANDIDATES))
        for candidate in ordered:
            started = time.perf_counter()
            samples = pointsets[candidate.correction_samples]
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                full_final = full_sample_final(mlp, samples)
                reduced_small, _ = structural_prediction(
                    mlp, samples, k2, alphas, {}, plans, identity, OPTIMIZED
                )
                corrected_final = fnp.add(
                    reduced_main[-1], full_final - reduced_small[-1]
                )
                prediction = fnp.concatenate(
                    (k2[:-1], corrected_final[None, :]), axis=0
                )
            flops_used = (
                k2_flops + merge_flops + base_flops + int(context.flops_used)
            )
            residual = (
                k2_residual
                + merge_residual
                + plan_time
                + base_residual
                + float(context.residual_wall_time_s or 0.0)
            )
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            diagnostics[candidate.name].append(base_diagnostics)
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
    prediction_arrays = {name: np.stack(values) for name, values in predictions.items()}
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
    crossfit = {
        candidate.name: {
            **{
                field: 0.5 * (
                    parity["0"][candidate.name][field]["mean"]
                    + parity["1"][candidate.name][field]["mean"]
                )
                for field in fields
            },
            "lambda_mean": 0.5 * (
                parity["0"][candidate.name]["lambda"]
                + parity["1"][candidate.name]["lambda"]
            ),
        }
        for candidate in CANDIDATES
    }
    result = {
        "schema_version": "1.0",
        "experiment": "hope_coupled_residual_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "budget": args.budget,
            "main_samples": MAIN_SAMPLES,
            "correction_samples": CORRECTION_SAMPLES,
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
        },
        "summary": {"parity_crossfit": parity, "crossfit_mean": crossfit},
        "per_mlp": [
            {
                "mlp_id": int(mlp_id),
                "compute": {
                    candidate.name: compute[candidate.name][index]
                    for candidate in CANDIDATES
                },
                "diagnostics": {
                    candidate.name: diagnostics[candidate.name][index]
                    for candidate in CANDIDATES
                },
            }
            for index, mlp_id in enumerate(ids_array)
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    stem = f"hope_coupled_residual_limit{args.limit_mlps}"
    json_path = RESULTS / f"{stem}.json"
    cache_path = RESULTS / f"{stem}.npz"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        cache_path,
        mlp_ids=ids_array,
        targets=targets_array,
        k2=k2_array,
        **{f"prediction__{name}": values for name, values in prediction_arrays.items()},
    )
    return json_path, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument(
        "--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    json_path, cache_path = run(args)
    print(f"wrote {json_path.relative_to(ROOT)} and {cache_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
