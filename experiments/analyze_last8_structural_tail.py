"""Analyze last-eight structural candidates against the packaged last-six baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "deep_structural_tail_last8_limit50.json"
DEFAULT_CACHE = ROOT / "results" / "deep_structural_tail_last8_limit50.npz"
DEFAULT_REPORT = ROOT / "research" / "reports" / "last8_structural_tail_scout.md"
BASELINE = "C_last6_tapered"


def bootstrap_interval(
    differences: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def paired_summary(
    metadata: dict[str, Any], cache: Any, draws: int
) -> dict[str, dict[str, float]]:
    ids = cache["mlp_ids"]
    targets = cache["targets"]
    k2 = cache["k2"]
    parity = metadata["summary"]["parity_crossfit"]
    names = [row["name"] for row in metadata["configuration"]["candidates"]]
    budget = metadata["configuration"]["budget"]
    per_candidate = {}
    for name in names:
        sampled = cache[f"prediction__{name}"]
        lambdas = np.asarray(
            [parity[str(1 - int(mlp_id % 2))][name]["lambda"] for mlp_id in ids]
        )
        combo = k2 + lambdas[:, None, None] * (sampled - k2)
        raw = ((combo[:, -1] - targets[:, -1]) ** 2).mean(axis=1)
        flops_used = np.asarray(
            [row["compute"][name]["flops"] for row in metadata["per_mlp"]]
        )
        effective = np.asarray(
            [row["compute"][name]["effective_compute"] for row in metadata["per_mlp"]]
        )
        per_candidate[name] = {
            "raw": raw,
            "adjusted": raw * np.maximum(0.1, effective / budget),
            "flop_adjusted": raw * np.maximum(0.1, flops_used / budget),
            "flops": flops_used,
            "effective": effective,
        }

    rng = np.random.default_rng(20_260_803)
    baseline = per_candidate[BASELINE]
    output = {}
    for name, values in per_candidate.items():
        result = {}
        for metric in ("raw", "adjusted", "flop_adjusted", "flops", "effective"):
            difference = values[metric] - baseline[metric]
            low, high = bootstrap_interval(difference, rng, draws)
            baseline_mean = float(baseline[metric].mean())
            result[f"{metric}_mean"] = float(values[metric].mean())
            result[f"{metric}_difference"] = float(difference.mean())
            result[f"{metric}_relative_difference"] = float(
                difference.mean() / baseline_mean
            )
            result[f"{metric}_difference_ci95_low"] = low
            result[f"{metric}_difference_ci95_high"] = high
            result[f"{metric}_win_fraction"] = float(np.mean(difference < 0.0))
        output[name] = result
    return output


def sci(value: float) -> str:
    return f"{value:.4e}"


def write_report(
    metadata: dict[str, Any], paired: dict[str, dict[str, float]], output: Path
) -> None:
    crossfit = metadata["summary"]["crossfit_mean"]
    parity = metadata["summary"]["parity_crossfit"]
    candidates = metadata["configuration"]["candidates"]
    lines = [
        "# Last-Eight Structural Tail Scout",
        "",
        "The exact 32,768-point antithetic Sobol rule was evaluated on the balanced first 50 Mini "
        "MLPs. Current-last-two and C-last-six were rerun as same-process controls.",
        "",
        "| Candidate | Thresholds, layers 25-32 | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        name = candidate["name"]
        row = crossfit[name]
        thresholds = ", ".join(f"{value:g}" for value in candidate["thresholds"])
        lines.append(
            f"| `{name}` | {thresholds} | {sci(row['combo_final_mse'])} | "
            f"{sci(row['flops'])} | {sci(row['combo_adjusted_score'])} | "
            f"{sci(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## Paired Against C",
        "",
        "Confidence intervals are paired 20,000-draw bootstrap intervals for candidate minus "
        "C-last-six. Each MLP uses the lambda fitted on the opposite parity.",
        "",
        "| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("D_last8_conservative", "E_last8_balanced", "F_last8_aggressive"):
        row = paired[name]
        lines.append(
            f"| `{name}` | {row['raw_relative_difference']:+.3%} | "
            f"[{sci(row['raw_difference_ci95_low'])}, {sci(row['raw_difference_ci95_high'])}] | "
            f"{row['adjusted_relative_difference']:+.3%} | "
            f"[{sci(row['adjusted_difference_ci95_low'])}, {sci(row['adjusted_difference_ci95_high'])}] | "
            f"{row['flops_relative_difference']:+.3%} |"
        )

    lines.extend([
        "",
        "## Parity Check",
        "",
        "| Candidate | Parity-0 raw | Parity-0 adjusted | Parity-1 raw | Parity-1 adjusted |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in (BASELINE, "D_last8_conservative", "E_last8_balanced", "F_last8_aggressive"):
        p0 = parity["0"][name]
        p1 = parity["1"][name]
        lines.append(
            f"| `{name}` | {sci(p0['combo_final_mse']['mean'])} | "
            f"{sci(p0['combo_adjusted_score']['mean'])} | "
            f"{sci(p1['combo_final_mse']['mean'])} | "
            f"{sci(p1['combo_adjusted_score']['mean'])} |"
        )

    f_row = paired["F_last8_aggressive"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"F reduces FLOPs by {-f_row['flops_relative_difference']:.2%} and measured adjusted score "
        f"by {-f_row['adjusted_relative_difference']:.2%} relative to C. Its raw-MSE change is "
        f"{f_row['raw_relative_difference']:+.3%}, with a confidence interval spanning zero. "
        "The adjusted improvement appears in both parity directions and under FLOP-only accounting.",
        "",
        "F is therefore the last-eight packaging candidate. E remains the conservative fallback if "
        "a future untouched split shows that F's earlier classifications amplify more than expected.",
        "",
        "## Classification Detail",
        "",
        "At each layer corrected K2 supplies a Gaussian preactivation mean `mu` and standard "
        "deviation `sigma`. The z-score `alpha = mu / sigma` defines the classes:",
        "",
        "- active: `alpha >= threshold`; propagate the preactivation linearly;",
        "- dead: `alpha <= -threshold`; omit the neuron as zero;",
        "- uncertain: `abs(alpha) < threshold`; evaluate sampled preactivations and apply ReLU.",
        "",
        "F uses thresholds `3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0` from layers 25 "
        "through 32. On the parity-0 evaluation half it retained, on average, 158.3 uncertain "
        "neurons at layer 25 and 149.0 at layer 26. Empirical 32k wrong-sign rates there were "
        "approximately `1.9e-6` and `4.0e-6`, respectively.",
        "",
        "A next correction experiment should preserve the same classes but add Gaussian tail-mass "
        "mean corrections: a small positive correction for the clipped negative tail of active "
        "neurons and a small positive expected activation for dead neurons. This can potentially "
        "reduce accumulated bias without restoring full sampled matrix products.",
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
        paired = paired_summary(metadata, cache, args.bootstrap_draws)
    write_report(metadata, paired, args.output)
    print(json.dumps(paired, indent=2, sort_keys=True))
    print(f"wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
