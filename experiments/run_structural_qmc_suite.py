"""Run reusable structural-QMC point-set comparisons on the first 50 MLPs."""
from __future__ import annotations

import argparse
import importlib.util
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
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, SCORE_FLOOR, as_mlp
from run_grid_combo import DEFAULT_SOBOL_DIRECTION_FILE, SOBOL_BITS, SOBOL_SCALE
from run_structural_tail_scout import (
    aggregate,
    antithetic_sobol_samples,
    sobol_base_integers,
)
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_ESTIMATOR = (
    ROOT / "submissions" / "structural_tail_sobol_32768" / "estimator.py"
)
SEED_OFFSET = 8_675_309


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    total_samples: int
    components: int = 1
    seed: int = 0


def load_structural_module() -> Any:
    spec = importlib.util.spec_from_file_location("qmc_structural_estimator", STRUCTURAL_ESTIMATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STRUCTURAL_ESTIMATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_candidates(stage: str) -> list[Candidate]:
    if stage == "count":
        return [
            Candidate(f"xor_n{n}", "xor", n)
            for n in (16_384, 20_480, 24_576, 28_672, 32_768)
        ]
    if stage == "ensemble":
        return [
            Candidate("xor_1x32768", "xor", 32_768, 1),
            Candidate("xor_2x16384", "xor", 32_768, 2),
            Candidate("xor_4x8192", "xor", 32_768, 4),
        ]
    if stage == "scramble":
        return [
            Candidate("xor_seed0", "xor", 32_768, seed=0),
            Candidate("xor_seed1", "xor", 32_768, seed=1),
            Candidate("lms_seed0", "lms", 32_768, seed=0),
            Candidate("lms_seed1", "lms", 32_768, seed=1),
            Candidate("owen_seed0", "owen", 32_768, seed=0),
            Candidate("owen_seed1", "owen", 32_768, seed=1),
        ]
    raise ValueError(f"unknown stage {stage!r}")


def _digital_shift(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    return rng.integers(0, 1 << SOBOL_BITS, size=width, dtype=np.uint32)


def _linear_matrix_scramble(integers: np.ndarray, seed: int) -> np.ndarray:
    """Apply a random nonsingular lower-triangular binary scramble."""
    rng = np.random.default_rng(seed + 31_415_927)
    output = np.zeros_like(integers, dtype=np.uint32)
    for row in range(SOBOL_BITS):
        bit_position = SOBOL_BITS - 1 - row
        diagonal = np.uint32(1 << bit_position)
        higher = np.uint32((0xFFFFFFFF << bit_position) & 0xFFFFFFFF) ^ diagonal
        random_bits = rng.integers(0, 1 << SOBOL_BITS, size=integers.shape[1], dtype=np.uint32)
        masks = np.bitwise_or(np.bitwise_and(random_bits, higher), diagonal)
        parity = np.bitwise_and(
            np.bitwise_count(np.bitwise_and(integers, masks[None, :])), 1
        ).astype(np.uint32)
        output = np.bitwise_or(output, np.left_shift(parity, bit_position))
    return np.bitwise_xor(output, _digital_shift(integers.shape[1], seed)[None, :])


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = np.bitwise_xor(values, np.right_shift(values, 30)) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = np.bitwise_xor(values, np.right_shift(values, 27)) * np.uint64(
        0x94D049BB133111EB
    )
    return np.bitwise_xor(values, np.right_shift(values, 31))


def _owen_scramble(integers: np.ndarray, seed: int) -> np.ndarray:
    """Apply a deterministic 32-bit nested uniform binary scramble."""
    source = integers.astype(np.uint64)
    dimensions = np.arange(integers.shape[1], dtype=np.uint64)[None, :]
    output = np.zeros_like(integers, dtype=np.uint32)
    for row in range(SOBOL_BITS):
        bit_position = SOBOL_BITS - 1 - row
        prefix = np.right_shift(source, bit_position + 1)
        key = np.bitwise_xor(prefix, dimensions * np.uint64(0x9E3779B97F4A7C15))
        row_key = np.uint64(
            (
                (seed + 1) * 0xD1B54A32D192ED03
                + row * 0x94D049BB133111EB
            )
            & 0xFFFFFFFFFFFFFFFF
        )
        key = np.bitwise_xor(key, row_key)
        random_bit = np.bitwise_and(_splitmix64(key), np.uint64(1)).astype(np.uint32)
        source_bit = np.bitwise_and(np.right_shift(integers, bit_position), 1)
        scrambled_bit = np.bitwise_xor(source_bit, random_bit)
        output = np.bitwise_or(output, np.left_shift(scrambled_bit, bit_position))
    return output


def _normal_antithetic(integers: np.ndarray) -> fnp.ndarray:
    uniforms = np.clip(
        (integers.astype(np.float64) + 0.5) / SOBOL_SCALE,
        1e-7,
        1.0 - 1e-7,
    )
    base = flops.stats.norm.ppf(fnp.array(uniforms, dtype=fnp.float64)).astype(fnp.float32)
    return fnp.concatenate((base, fnp.negative(base)), axis=0)


def pointset_component(
    method: str,
    n_samples: int,
    width: int,
    seed: int,
    direction_file: Path,
) -> fnp.ndarray:
    base = sobol_base_integers(n_samples, width, direction_file, "numpy")
    if method == "xor":
        return antithetic_sobol_samples(base, seed, SEED_OFFSET, "numpy")
    integers = np.asarray(base, dtype=np.uint32)
    if method == "lms":
        integers = _linear_matrix_scramble(integers, seed)
    elif method == "owen":
        integers = _owen_scramble(integers, seed)
    else:
        raise ValueError(f"unknown point-set method {method!r}")
    return _normal_antithetic(integers)


def build_pointset(candidate: Candidate, width: int, direction_file: Path) -> fnp.ndarray:
    if candidate.total_samples % candidate.components:
        raise ValueError(f"{candidate.name}: sample count is not divisible by components")
    component_samples = candidate.total_samples // candidate.components
    parts = [
        pointset_component(
            candidate.method,
            component_samples,
            width,
            candidate.seed + component,
            direction_file,
        )
        for component in range(candidate.components)
    ]
    return parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=0)


def fit_lambda(
    mlp_ids: np.ndarray,
    k2: np.ndarray,
    structural: np.ndarray,
    targets: np.ndarray,
    parity: int,
) -> float:
    selected = mlp_ids % 2 == parity
    delta = structural[selected, -1] - k2[selected, -1]
    baseline_error = k2[selected, -1] - targets[selected, -1]
    denominator = float(np.vdot(delta, delta))
    value = -float(np.vdot(delta, baseline_error)) / denominator if denominator else 0.0
    return min(1.0, max(0.0, value))


def summarize_candidate(
    mlp_ids: np.ndarray,
    k2: np.ndarray,
    structural: np.ndarray,
    targets: np.ndarray,
    compute_rows: list[dict[str, float | int]],
    calibration_parity: int,
    budget: int,
) -> dict[str, Any]:
    lambda_ = fit_lambda(mlp_ids, k2, structural, targets, calibration_parity)
    selected = mlp_ids % 2 != calibration_parity
    direct = structural[selected]
    combo = k2[selected] + lambda_ * (direct - k2[selected])
    target = targets[selected]
    direct_error = (direct - target) ** 2
    combo_error = (combo - target) ** 2
    eval_compute = [row for row, keep in zip(compute_rows, selected) if keep]
    effective = np.asarray([row["effective_compute"] for row in eval_compute], dtype=np.float64)
    flops_used = np.asarray([row["flops"] for row in eval_compute], dtype=np.float64)
    final_per_mlp = combo_error[:, -1].mean(axis=1)
    return {
        "lambda": lambda_,
        "evaluation_mlps": int(selected.sum()),
        "direct_final_mse": aggregate(direct_error[:, -1].mean(axis=1)),
        "combo_final_mse": aggregate(final_per_mlp),
        "combo_all_layer_mse": aggregate(combo_error.mean(axis=(1, 2))),
        "combo_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, effective / budget)
        ),
        "combo_flop_only_adjusted_score": aggregate(
            final_per_mlp * np.maximum(SCORE_FLOOR, flops_used / budget)
        ),
        "flops": aggregate(flops_used),
        "effective_compute": aggregate(effective),
        "residual_wall_time_s": aggregate(
            row["residual_wall_time_s"] for row in eval_compute
        ),
        "runtime_s": aggregate(row["runtime_s"] for row in eval_compute),
        "budget_exceeded_fraction": float(np.mean(effective > budget)),
    }


def diagnostic_row(alpha1: np.ndarray, alpha2: np.ndarray) -> dict[str, float]:
    normalizer = 1.0 / math.sqrt(2.0 * math.pi)
    return {
        "boundary_mass_1": float(np.mean(normalizer * np.exp(-0.5 * alpha1**2))),
        "boundary_mass_2": float(np.mean(normalizer * np.exp(-0.5 * alpha2**2))),
        "uncertain_fraction_1": float(np.mean(np.abs(alpha1) < 3.0)),
        "uncertain_fraction_2": float(np.mean(np.abs(alpha2) < 3.0)),
        "mean_abs_alpha_1": float(np.mean(np.abs(alpha1))),
        "mean_abs_alpha_2": float(np.mean(np.abs(alpha2))),
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))[
        : args.limit_mlps
    ]
    candidates = stage_candidates(args.stage)
    module = load_structural_module()
    width = as_mlp(rows[0]).width
    identity = fnp.array(np.eye(width, dtype=np.float32))

    pointsets: dict[str, fnp.ndarray] = {}
    setup_rows = {}
    for candidate in candidates:
        started = time.perf_counter()
        pointsets[candidate.name] = build_pointset(candidate, width, args.sobol_direction_file)
        setup_rows[candidate.name] = {
            "runtime_s": time.perf_counter() - started,
            "shape": list(pointsets[candidate.name].shape),
            "bytes": int(pointsets[candidate.name].size * pointsets[candidate.name].itemsize),
        }
        print(f"built {candidate.name}", flush=True)

    mlp_ids = []
    targets = []
    k2_predictions = []
    alpha1_rows = []
    alpha2_rows = []
    diagnostics = []
    predictions: dict[str, list[np.ndarray]] = {candidate.name: [] for candidate in candidates}
    compute: dict[str, list[dict[str, float | int]]] = {
        candidate.name: [] for candidate in candidates
    }

    for row_index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        mlp_id = int(row["mlp_id"])
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
            k2, alphas = module._gaussian_classifier(mlp)
        k2_residual = float(k2_context.residual_wall_time_s or 0.0)
        alpha1 = np.asarray(alphas[0], dtype=np.float64)
        alpha2 = np.asarray(alphas[1], dtype=np.float64)
        ordered = candidates if (mlp_id // 2) % 2 == 0 else list(reversed(candidates))
        for candidate in ordered:
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
                prediction = module._structural_prediction(
                    mlp,
                    pointsets[candidate.name],
                    alphas[0],
                    alphas[1],
                    identity,
                )
            runtime_s = time.perf_counter() - started
            residual_s = k2_residual + float(context.residual_wall_time_s or 0.0)
            flops_used = int(k2_context.flops_used) + int(context.flops_used)
            predictions[candidate.name].append(np.asarray(prediction, dtype=np.float64))
            compute[candidate.name].append(
                {
                    "flops": flops_used,
                    "residual_wall_time_s": residual_s,
                    "effective_compute": flops_used + LAMBDA_FLOPS_PER_SECOND * residual_s,
                    "runtime_s": runtime_s,
                }
            )

        mlp_ids.append(mlp_id)
        targets.append(np.asarray(row["all_layer_means"], dtype=np.float64))
        k2_predictions.append(np.asarray(k2, dtype=np.float64))
        alpha1_rows.append(alpha1)
        alpha2_rows.append(alpha2)
        diagnostics.append(diagnostic_row(alpha1, alpha2))
        print(f"{row_index}/{len(rows)} {row['mlp_name']} completed", flush=True)

    ids_array = np.asarray(mlp_ids, dtype=np.int64)
    targets_array = np.stack(targets)
    k2_array = np.stack(k2_predictions)
    prediction_arrays = {
        name: np.stack(candidate_predictions)
        for name, candidate_predictions in predictions.items()
    }
    summaries = {
        candidate.name: {
            "configuration": asdict(candidate),
            **summarize_candidate(
                ids_array,
                k2_array,
                prediction_arrays[candidate.name],
                targets_array,
                compute[candidate.name],
                args.calibration_parity,
                args.budget,
            ),
        }
        for candidate in candidates
    }

    result = {
        "schema_version": "1.0",
        "experiment": "structural_qmc_suite",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "calibration_parity": args.calibration_parity,
            "budget": args.budget,
            "balanced_order": True,
            "sobol_direction_file": str(args.sobol_direction_file),
            "candidates": [asdict(candidate) for candidate in candidates],
        },
        "setup": setup_rows,
        "summary": summaries,
        "per_mlp": [
            {
                "mlp_id": int(mlp_id),
                "diagnostics": diagnostic,
                "compute": {
                    candidate.name: compute[candidate.name][index]
                    for candidate in candidates
                },
            }
            for index, (mlp_id, diagnostic) in enumerate(zip(mlp_ids, diagnostics))
        ],
    }

    RESULTS.mkdir(exist_ok=True)
    json_path = RESULTS / f"structural_qmc_{args.stage}_limit{args.limit_mlps}.json"
    cache_path = RESULTS / f"structural_qmc_{args.stage}_limit{args.limit_mlps}.npz"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        cache_path,
        mlp_ids=ids_array,
        targets=targets_array,
        k2=k2_array,
        alpha1=np.stack(alpha1_rows),
        alpha2=np.stack(alpha2_rows),
        **{f"prediction__{name}": value for name, value in prediction_arrays.items()},
    )
    return json_path, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("count", "ensemble", "scramble"), required=True)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--budget", type=int, default=BUDGET)
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
