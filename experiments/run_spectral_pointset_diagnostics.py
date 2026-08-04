"""Fourier/Walsh diagnostics for 4k sampling-rule design.

This script is offline research only. It tests whether final-layer random-MLP
outputs have stable spectral structure when ordered by a candidate point set:

* Rank-1 lattice rules are summarized by ordinary FFT energy.
* Sobol/digital-net rules are summarized by a fast Walsh-Hadamard transform.

The goal is Stage A from the Fourier/Walsh experiment brief: before building a
new estimator, check whether spectral energy is nonuniform and stable across a
50-MLP development slice. If it is not, there is no fixed 4k rule to learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from run_baselines import DATASET, RESULTS, REVISION, as_mlp, git_value  # noqa: E402
from run_grid_combo import (  # noqa: E402
    DEFAULT_RQMC_MULTIPLIER,
    DEFAULT_RQMC_SEED_OFFSET,
    DEFAULT_SOBOL_DIRECTION_FILE,
    lattice_samples,
    sobol_samples,
)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _rademacher_projection_matrix(width: int, n_random: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    columns = [np.ones(width, dtype=np.float64) / math.sqrt(width)]
    labels = ["ones"]
    selected_neurons = [0, width // 4, width // 2, (3 * width) // 4, width - 1]
    for neuron in selected_neurons:
        vector = np.zeros(width, dtype=np.float64)
        vector[neuron] = 1.0
        columns.append(vector)
        labels.append(f"neuron_{neuron}")
    for index in range(n_random):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=width)
        columns.append(signs / math.sqrt(width))
        labels.append(f"rademacher_{index}")
    return np.stack(columns, axis=1), labels


def _forward_final_outputs(mlp: Any, x: fnp.ndarray) -> np.ndarray:
    activations = x
    for w in mlp.weights:
        activations = fnp.maximum(fnp.matmul(activations, w), 0.0)
    return np.asarray(activations, dtype=np.float64)


def _centered_projection_values(outputs: np.ndarray, projections: np.ndarray) -> np.ndarray:
    values = outputs @ projections
    values = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return values / np.maximum(scale, 1e-12)


def _fft_power(values: np.ndarray) -> np.ndarray:
    coeffs = np.fft.fft(values, axis=0) / values.shape[0]
    power = np.mean(np.abs(coeffs) ** 2, axis=1)
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0.0:
        return power
    return power / total


def _fwht(values: np.ndarray) -> np.ndarray:
    transformed = values.copy()
    n = transformed.shape[0]
    step = 1
    while step < n:
        block = step * 2
        for start in range(0, n, block):
            left = transformed[start : start + step].copy()
            right = transformed[start + step : start + block].copy()
            transformed[start : start + step] = left + right
            transformed[start + step : start + block] = left - right
        step = block
    return transformed


def _walsh_power(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    if n & (n - 1):
        raise ValueError("Walsh transform requires a power-of-two sample count")
    coeffs = _fwht(values) / n
    power = np.mean(coeffs * coeffs, axis=1)
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0.0:
        return power
    return power / total


def _band_edges(n: int) -> list[tuple[int, int, str]]:
    edges = []
    start = 1
    while start < n:
        end = min(n, start * 2)
        edges.append((start, end, f"{start}-{end - 1}"))
        start = end
    return edges


def _spectral_summary(profile: np.ndarray, top_k: int) -> dict[str, Any]:
    nonzero = profile[1:] if len(profile) > 1 else profile
    positive = nonzero[nonzero > 0.0]
    if len(positive) == 0:
        entropy = 0.0
        effective_bins = 0.0
    else:
        entropy = float(-(positive * np.log(positive)).sum())
        effective_bins = float(math.exp(entropy))
    order = np.argsort(profile)[::-1]
    top = [
        {"index": int(index), "energy": float(profile[index])}
        for index in order[:top_k]
        if index != 0 and profile[index] > 0.0
    ]
    return {
        "top_bins": top,
        "top_8_energy": float(np.sort(profile)[-8:].sum()) if len(profile) >= 8 else float(profile.sum()),
        "top_32_energy": float(np.sort(profile)[-32:].sum()) if len(profile) >= 32 else float(profile.sum()),
        "effective_bins": effective_bins,
        "effective_fraction": effective_bins / max(1, len(nonzero)),
        "max_bin_energy": float(profile[order[0]]) if len(order) else 0.0,
        "dyadic_bands": [
            {"band": label, "energy": float(profile[start:end].sum())}
            for start, end, label in _band_edges(len(profile))
        ],
    }


def _profile_stability(profiles: list[np.ndarray]) -> dict[str, float | None]:
    if len(profiles) < 2:
        return {"mean_pairwise_corr": None, "median_pairwise_corr": None}
    stacked = np.stack(profiles, axis=0)
    centered = stacked - stacked.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    corrs = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            denom = norms[i] * norms[j]
            if denom > 0.0:
                corrs.append(float(np.dot(centered[i], centered[j]) / denom))
    if not corrs:
        return {"mean_pairwise_corr": None, "median_pairwise_corr": None}
    return {
        "mean_pairwise_corr": float(np.mean(corrs)),
        "median_pairwise_corr": float(np.median(corrs)),
    }


def _generator_powers(width: int, n_samples: int, multiplier: int) -> np.ndarray:
    powers = np.empty(width, dtype=np.int64)
    value = 1
    modulus = max(2, n_samples)
    for index in range(width):
        powers[index] = value
        value = (value * multiplier) % modulus
        if value == 0:
            value = 1
    return powers


def _candidate_alias_penalty(
    energy: np.ndarray,
    old_n: int,
    candidate_n: int,
    width: int,
    multiplier: int,
    max_frequency: int,
) -> dict[str, float]:
    """Cheap proxy for candidate lattice dual risk using one-sparse aliases.

    This is intentionally conservative: it only scores frequencies that are
    visible as one-dimensional lattice-order bins and dual to a candidate
    rank-1 lattice. It is a Stage-B scout, not a full CBC objective.
    """
    powers = _generator_powers(width, candidate_n, multiplier)
    penalty = 0.0
    hits = 0
    usable = min(max_frequency, old_n - 1)
    for dim_power in powers:
        for freq in range(1, usable + 1):
            if (freq * int(dim_power)) % candidate_n == 0:
                bin_index = int((freq * int(dim_power)) % old_n)
                if bin_index:
                    penalty += float(energy[bin_index])
                    hits += 1
    return {"penalty": penalty, "hits": float(hits)}


def _evaluate_pointset(
    row: dict[str, Any],
    projections: np.ndarray,
    args: argparse.Namespace,
    pointset: str,
    n_samples: int,
    seed_offset: int,
) -> np.ndarray:
    mlp = as_mlp(row)
    if pointset == "lattice":
        samples = lattice_samples(
            n_samples=n_samples,
            width=mlp.width,
            seed=mlp.seed,
            seed_offset=seed_offset,
            multiplier=args.multiplier,
        )
    elif pointset == "sobol":
        samples = sobol_samples(
            n_samples=n_samples,
            width=mlp.width,
            seed=mlp.seed,
            seed_offset=seed_offset,
            direction_file=args.sobol_direction_file,
        )
    else:
        raise ValueError(f"unknown pointset: {pointset}")
    outputs = _forward_final_outputs(mlp, samples)
    return _centered_projection_values(outputs, projections)


def _run_family_spectrum(
    rows: list[dict[str, Any]],
    projections: np.ndarray,
    args: argparse.Namespace,
    pointset: str,
    n_samples: int,
    transform_name: str,
) -> dict[str, Any]:
    profiles = []
    failures = []
    transform = _fft_power if transform_name == "fft" else _walsh_power
    seed_offsets = [args.seed_offset + args.shift_stride * shift for shift in range(args.n_shifts)]
    started = time.perf_counter()
    for row_index, row in enumerate(rows, start=1):
        for shift_index, seed_offset in enumerate(seed_offsets):
            try:
                values = _evaluate_pointset(row, projections, args, pointset, n_samples, seed_offset)
                profiles.append(transform(values))
                print(
                    f"{pointset}:{transform_name} {row_index}/{len(rows)} "
                    f"{row['mlp_name']} shift={shift_index} done",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "mlp_id": int(row["mlp_id"]),
                        "mlp_name": str(row["mlp_name"]),
                        "shift_index": shift_index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"{pointset}:{transform_name} {row_index}/{len(rows)} "
                    f"{row['mlp_name']} shift={shift_index} failed",
                    flush=True,
                )
    mean_profile = np.mean(np.stack(profiles, axis=0), axis=0) if profiles else np.zeros(n_samples)
    result = {
        "pointset": pointset,
        "transform": transform_name,
        "n_samples": n_samples,
        "n_profiles": len(profiles),
        "n_failures": len(failures),
        "runtime_s": time.perf_counter() - started,
        "summary": _spectral_summary(mean_profile, args.top_bins),
        "stability": _profile_stability(profiles),
        "mean_profile_head": [float(value) for value in mean_profile[: min(len(mean_profile), args.profile_head)]],
        "failures": failures,
    }
    if pointset == "lattice" and args.candidate_multipliers:
        result["candidate_alias_penalties"] = {
            str(multiplier): _candidate_alias_penalty(
                mean_profile,
                old_n=n_samples,
                candidate_n=args.candidate_n_samples,
                width=args.width,
                multiplier=multiplier,
                max_frequency=args.alias_max_frequency,
            )
            for multiplier in args.candidate_multipliers
        }
    return result


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    if not rows:
        raise SystemExit("no rows selected")
    args.width = int(np.asarray(rows[0]["weights"]).shape[1])
    projections, projection_labels = _rademacher_projection_matrix(
        args.width, args.random_projections, args.projection_seed
    )
    analyses = []
    if args.run_lattice:
        analyses.append(
            _run_family_spectrum(rows, projections, args, "lattice", args.lattice_samples, "fft")
        )
    if args.run_sobol:
        if args.sobol_samples & (args.sobol_samples - 1):
            raise SystemExit("--sobol-samples must be a power of two")
        analyses.append(
            _run_family_spectrum(rows, projections, args, "sobol", args.sobol_samples, "walsh")
        )

    result = {
        "schema_version": "1.0",
        "experiment": "spectral_pointset_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "configuration": {
            "phase": "phase1",
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_split": "mini",
            "limit_mlps": args.limit_mlps,
            "n_shifts": args.n_shifts,
            "seed_offset": args.seed_offset,
            "shift_stride": args.shift_stride,
            "lattice_samples": args.lattice_samples if args.run_lattice else None,
            "sobol_samples": args.sobol_samples if args.run_sobol else None,
            "multiplier": args.multiplier,
            "candidate_n_samples": args.candidate_n_samples,
            "candidate_multipliers": args.candidate_multipliers,
            "alias_max_frequency": args.alias_max_frequency,
            "projection_labels": projection_labels,
            "sobol_direction_file": str(args.sobol_direction_file) if args.run_sobol else None,
            "sobol_direction_sha256": (
                hashlib.sha256(args.sobol_direction_file.read_bytes()).hexdigest()
                if args.run_sobol and args.sobol_direction_file.exists()
                else None
            ),
        },
        "source": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "python": sys.version,
            "whestbench": getattr(whestbench, "__version__", None),
        },
        "analyses": analyses,
        "decision_hints": [
            "Continue to Stage B only if top-bin energy is high and pairwise profile correlation is positive.",
            "Flat spectra or unstable top bins mean a fixed 4k point set is unlikely to help.",
            "For lattice, ordinary FFT bins diagnose rank-1 lattice order; for Sobol, Walsh bins diagnose digital-net order.",
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    modes = []
    if args.run_lattice:
        modes.append(f"lattice{args.lattice_samples}")
    if args.run_sobol:
        modes.append(f"sobol{args.sobol_samples}")
    suffix = f"limit{args.limit_mlps or 'all'}_shifts{args.n_shifts}_{'_'.join(modes)}"
    output = RESULTS / f"spectral_pointset_diagnostics_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--n-shifts", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_RQMC_SEED_OFFSET)
    parser.add_argument("--shift-stride", type=int, default=1_000_003)
    parser.add_argument("--lattice-samples", type=int, default=4096)
    parser.add_argument("--sobol-samples", type=int, default=4096)
    parser.add_argument("--multiplier", type=int, default=DEFAULT_RQMC_MULTIPLIER)
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
    parser.add_argument("--random-projections", type=int, default=8)
    parser.add_argument("--projection-seed", type=int, default=12345)
    parser.add_argument("--top-bins", type=int, default=16)
    parser.add_argument("--profile-head", type=int, default=64)
    parser.add_argument("--candidate-n-samples", type=int, default=4096)
    parser.add_argument(
        "--candidate-multipliers",
        type=int,
        nargs="*",
        default=[17_797, 2_891_336_453, 50_933, 117_917, 1_048_573],
    )
    parser.add_argument("--alias-max-frequency", type=int, default=4096)
    parser.add_argument("--skip-lattice", dest="run_lattice", action="store_false")
    parser.add_argument("--skip-sobol", dest="run_sobol", action="store_false")
    parser.set_defaults(run_lattice=True, run_sobol=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 0:
        raise SystemExit("--limit-mlps must be positive")
    if args.n_shifts <= 0:
        raise SystemExit("--n-shifts must be positive")
    if args.lattice_samples <= 1 or args.sobol_samples <= 1:
        raise SystemExit("sample counts must be greater than one")
    if args.run_sobol and not args.sobol_direction_file.exists():
        raise SystemExit(f"Sobol direction file not found: {args.sobol_direction_file}")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
