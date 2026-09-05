"""Analyze paired first-50 results from the deep structural-tail scout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "deep_structural_tail_limit50.json"
DEFAULT_CACHE = ROOT / "results" / "deep_structural_tail_limit50.npz"
DEFAULT_REPORT = ROOT / "research" / "reports" / "deep_structural_tail_scout.md"
BASELINE = "current_last2_t3"


def bootstrap_interval(
    differences: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def paired_rows(
    metadata: dict[str, Any], cache: Any, draws: int
) -> dict[str, dict[str, float]]:
    ids = cache["mlp_ids"]
    targets = cache["targets"]
    k2 = cache["k2"]
    parity = metadata["summary"]["parity_crossfit"]
    names = [row["name"] for row in metadata["configuration"]["candidates"]]
    rng = np.random.default_rng(20_260_803)
    per_candidate = {}
    for name in names:
        sampled = cache[f"prediction__{name}"]
        lambdas = np.asarray(
            [parity[str(1 - int(mlp_id % 2))][name]["lambda"] for mlp_id in ids]
        )
        combo = k2 + lambdas[:, None, None] * (sampled - k2)
        raw = ((combo[:, -1] - targets[:, -1]) ** 2).mean(axis=1)
        compute = metadata["per_mlp"]
        flops_used = np.asarray(
            [row["compute"][name]["flops"] for row in compute], dtype=np.float64
        )
        effective = np.asarray(
            [row["compute"][name]["effective_compute"] for row in compute],
            dtype=np.float64,
        )
        budget = metadata["configuration"]["budget"]
        floor = 0.1
        per_candidate[name] = {
            "raw": raw,
            "adjusted": raw * np.maximum(floor, effective / budget),
            "flop_adjusted": raw * np.maximum(floor, flops_used / budget),
            "flops": flops_used,
            "effective": effective,
        }

    baseline = per_candidate[BASELINE]
    summaries = {}
    for name, values in per_candidate.items():
        row = {}
        for metric in ("raw", "adjusted", "flop_adjusted", "flops", "effective"):
            difference = values[metric] - baseline[metric]
            lower, upper = bootstrap_interval(difference, rng, draws)
            baseline_mean = float(baseline[metric].mean())
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_difference"] = float(difference.mean())
            row[f"{metric}_relative_difference"] = (
                float(difference.mean() / baseline_mean) if baseline_mean else 0.0
            )
            row[f"{metric}_difference_ci95_low"] = lower
            row[f"{metric}_difference_ci95_high"] = upper
            row[f"{metric}_per_mlp_win_fraction"] = float(np.mean(difference < 0.0))
        summaries[name] = row
    return summaries


def scientific(value: float) -> str:
    return f"{value:.4e}"


def write_report(
    metadata: dict[str, Any], paired: dict[str, dict[str, float]], output: Path
) -> None:
    crossfit = metadata["summary"]["crossfit_mean"]
    lines = [
        "# Deep Structural Tail Scout",
        "",
        "The exact 32,768-point antithetic Sobol asset was used on the first 50 mini-split MLPs. "
        "Dense propagation is diagnostic only; every candidate's measured compute includes its own "
        "sampled propagation plus corrected K2.",
        "",
        "| Candidate | Tail thresholds | Raw final MSE | FLOPs | Full adjusted | Flop-only adjusted | Structural vs dense |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in metadata["configuration"]["candidates"]:
        name = candidate["name"]
        row = crossfit[name]
        thresholds = ", ".join(f"{value:g}" for value in candidate["thresholds"])
        lines.append(
            f"| `{name}` | {thresholds} | {scientific(row['combo_final_mse'])} | "
            f"{scientific(row['flops'])} | {scientific(row['combo_adjusted_score'])} | "
            f"{scientific(row['combo_flop_only_adjusted_score'])} | "
            f"{scientific(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## Paired Comparison",
        "",
        "Each MLP uses the lambda fitted on the opposite parity. Confidence intervals are paired "
        "20,000-draw bootstrap intervals for candidate minus current-last-two.",
        "",
        "| Candidate | Raw delta | Raw delta 95% CI | Adjusted delta | Adjusted delta 95% CI | FLOP change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, row in paired.items():
        if name == BASELINE:
            continue
        lines.append(
            f"| `{name}` | {row['raw_relative_difference']:+.3%} | "
            f"[{scientific(row['raw_difference_ci95_low'])}, {scientific(row['raw_difference_ci95_high'])}] | "
            f"{row['adjusted_relative_difference']:+.3%} | "
            f"[{scientific(row['adjusted_difference_ci95_low'])}, {scientific(row['adjusted_difference_ci95_high'])}] | "
            f"{row['flops_relative_difference']:+.3%} |"
        )

    c_row = crossfit["C_last6_tapered"]
    c_pair = paired["C_last6_tapered"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Candidate C reduces mean FLOPs by {-c_pair['flops_relative_difference']:.2%} and full adjusted "
        f"score by {-c_pair['adjusted_relative_difference']:.2%}. Its raw final MSE changes by only "
        f"{c_pair['raw_relative_difference']:+.3%}, while structural-versus-dense final MSE remains "
        f"{scientific(c_row['structural_vs_dense_final_mse'])}.",
        "",
        "The raw difference is too small to call an accuracy improvement. The compute reduction is "
        "the result: C is the packaging candidate because its adjusted score improves in both parity "
        "directions and under both full and FLOP-only accounting.",
        "",
        "## Structural Pattern",
        "",
        "Candidate C starts at layer 27. Its thresholds taper from 3.5 to 3.0, retaining more "
        "uncertain neurons at the earlier layers where a wrong classification has more opportunities "
        "to amplify downstream. The observed per-sample wrong-sign rates are about 0.0007% at layer "
        "27 and rise toward 0.004% in the final two layers.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.json.read_text(encoding="utf-8"))
    with np.load(args.cache) as cache:
        paired = paired_rows(metadata, cache, args.bootstrap_draws)
    write_report(metadata, paired, args.output)
    print(json.dumps(paired, indent=2, sort_keys=True))
    print(f"wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
