"""Paired analysis for sparse/pilot/reinvestment scout result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from run_baselines import BUDGET, SCORE_FLOOR


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def load_vectors(
    json_path: Path,
) -> tuple[dict[str, Any], np.lib.npyio.NpzFile, dict[str, dict[str, np.ndarray]]]:
    result = json.loads(json_path.read_text(encoding="utf-8"))
    cache = np.load(json_path.with_suffix(".npz"))
    ids = cache["mlp_ids"]
    targets = cache["targets"]
    k2 = cache["k2"]
    budget = int(result["configuration"].get("budget", BUDGET))
    vectors = {}
    for candidate in result["summary"]["crossfit_mean"]:
        sampled = cache[f"prediction__{candidate}"]
        lambdas = np.asarray([
            result["summary"]["parity_crossfit"][str(1 - int(mlp_id % 2))][
                candidate
            ]["lambda"]
            for mlp_id in ids
        ])
        combo = k2 + lambdas[:, None, None] * (sampled - k2)
        raw = ((combo[:, -1] - targets[:, -1]) ** 2).mean(axis=1)
        compute = np.asarray([
            row["compute"][candidate]["effective_compute"]
            for row in result["per_mlp"]
        ])
        flops = np.asarray([
            row["compute"][candidate]["flops"] for row in result["per_mlp"]
        ])
        vectors[candidate] = {
            "raw_mse": raw,
            "adjusted_score": raw * np.maximum(SCORE_FLOOR, compute / budget),
            "flop_only_score": raw * np.maximum(SCORE_FLOOR, flops / budget),
            "effective_compute": compute,
            "flops": flops,
        }
    return result, cache, vectors


def analyze(
    json_path: Path,
    control: str,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    result, cache, vectors = load_vectors(json_path)
    try:
        reference = vectors[control]
    except KeyError as error:
        raise SystemExit(f"unknown control {control!r}") from error

    rng = np.random.default_rng(seed)
    comparisons = {}
    for candidate, candidate_vectors in vectors.items():
        fields = {}
        for field, values in candidate_vectors.items():
            delta = values - reference[field]
            low, high = bootstrap_mean_interval(delta, rng, bootstrap_draws)
            reference_mean = float(reference[field].mean())
            fields[field] = {
                "mean": float(values.mean()),
                "mean_delta": float(delta.mean()),
                "mean_delta_percent": (
                    100.0 * float(delta.mean()) / reference_mean
                    if reference_mean
                    else None
                ),
                "bootstrap_95_delta": [low, high],
                "win_fraction": float(np.mean(values < reference[field])),
            }
        comparisons[candidate] = fields

    cache.close()
    return {
        "source": str(json_path),
        "stage": result["stage"],
        "control": control,
        "mlps": len(result["per_mlp"]),
        "bootstrap_draws": bootstrap_draws,
        "comparisons": comparisons,
    }


def print_summary(analysis: dict[str, Any]) -> None:
    print(f"control: {analysis['control']} ({analysis['mlps']} MLPs)")
    header = (
        f"{'candidate':36} {'raw delta %':>12} {'raw wins':>10} "
        f"{'adjusted delta %':>18} {'adjusted 95% CI':>27}"
    )
    print(header)
    for candidate, fields in analysis["comparisons"].items():
        raw = fields["raw_mse"]
        adjusted = fields["adjusted_score"]
        low, high = adjusted["bootstrap_95_delta"]
        print(
            f"{candidate:36} {raw['mean_delta_percent']:12.4f} "
            f"{raw['win_fraction']:10.3f} "
            f"{adjusted['mean_delta_percent']:18.4f} "
            f"[{low:.3e}, {high:.3e}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(
        args.result,
        args.control,
        args.bootstrap_draws,
        args.seed,
    )
    print_summary(analysis)
    if args.write_json:
        args.write_json.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
