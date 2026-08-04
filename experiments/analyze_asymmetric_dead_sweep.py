"""Analyze asymmetric dead-only classification against the symmetric last-16 control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "structural_asymmetric_dead_limit50.json"
DEFAULT_CACHE = ROOT / "results" / "structural_asymmetric_dead_limit50.npz"
DEFAULT_REPORT = ROOT / "research" / "reports" / "asymmetric_dead_sweep.md"
BASELINE = "M_last16_symmetric_control"


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
    dense = cache["dense"]
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
        structural = ((sampled[:, -1] - dense[:, -1]) ** 2).mean(axis=1)
        flops_used = np.asarray(
            [row["compute"][name]["flops"] for row in metadata["per_mlp"]]
        )
        effective = np.asarray(
            [row["compute"][name]["effective_compute"] for row in metadata["per_mlp"]]
        )
        per_candidate[name] = {
            "raw": raw,
            "structural": structural,
            "adjusted": raw * np.maximum(0.1, effective / budget),
            "flop_adjusted": raw * np.maximum(0.1, flops_used / budget),
            "flops": flops_used,
            "effective": effective,
        }

    rng = np.random.default_rng(20_260_803)
    baseline = per_candidate[BASELINE]
    output = {}
    for name, values in per_candidate.items():
        row = {}
        for metric in (
            "raw", "structural", "adjusted", "flop_adjusted", "flops", "effective"
        ):
            difference = values[metric] - baseline[metric]
            low, high = bootstrap_interval(difference, rng, draws)
            baseline_mean = float(baseline[metric].mean())
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_difference"] = float(difference.mean())
            row[f"{metric}_relative_difference"] = float(
                difference.mean() / baseline_mean
            )
            row[f"{metric}_difference_ci95_low"] = low
            row[f"{metric}_difference_ci95_high"] = high
            row[f"{metric}_win_fraction"] = float(np.mean(difference < 0.0))
        output[name] = row
    return output


def mean_class_counts(
    metadata: dict[str, Any], name: str
) -> dict[int, dict[str, float]]:
    layers = range(17, 33)
    return {
        layer: {
            group: float(np.mean([
                row["class_counts"][name][str(layer)][group]
                for row in metadata["per_mlp"]
            ]))
            for group in ("active", "dead", "uncertain")
        }
        for layer in layers
    }


def sci(value: float) -> str:
    return f"{value:.4e}"


def write_report(
    metadata: dict[str, Any], paired: dict[str, dict[str, float]], output: Path
) -> None:
    summary = metadata["summary"]["crossfit_mean"]
    parity = metadata["summary"]["parity_crossfit"]
    candidates = metadata["configuration"]["candidates"]
    mechanisms = {
        "M_last16_symmetric_control": "symmetric control",
        "N_last16_dead_only_same": "dead-only, same thresholds",
        "O_last16_dead_only_m025": "dead-only, threshold -0.25",
        "P_last16_dead_only_m050": "dead-only, threshold -0.50",
        "Q_last16_early_dead_m075": "dead-only -0.75 on layers 17-20",
    }
    lines = [
        "# Asymmetric Dead Sweep",
        "",
        "All candidates use corrected K2 plus the fixed 32,768-point antithetic Sobol asset. "
        "They were evaluated in balanced order on the first 50 Mini MLPs with opposite-parity "
        "lambda fitting. Dense sampling is diagnostic only.",
        "",
        "## Results",
        "",
        "| Candidate | Mechanism | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        name = candidate["name"]
        row = summary[name]
        lines.append(
            f"| `{name}` | {mechanisms[name]} | {sci(row['combo_final_mse'])} | "
            f"{sci(row['flops'])} | {sci(row['combo_adjusted_score'])} | "
            f"{sci(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## Paired Against M",
        "",
        "Intervals are paired 20,000-draw bootstrap intervals for candidate minus M. "
        "Negative changes are improvements.",
        "",
        "| Candidate | Raw change | Raw delta 95% CI | Structural change | FLOP change | Adjusted change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in list(mechanisms)[1:]:
        row = paired[name]
        lines.append(
            f"| `{name}` | {row['raw_relative_difference']:+.3%} | "
            f"[{sci(row['raw_difference_ci95_low'])}, {sci(row['raw_difference_ci95_high'])}] | "
            f"{row['structural_relative_difference']:+.2%} | "
            f"{row['flops_relative_difference']:+.3%} | "
            f"{row['adjusted_relative_difference']:+.3%} |"
        )

    lines.extend([
        "",
        "## Parity Check",
        "",
        "| Candidate | Parity-0 raw | Parity-1 raw | Parity-0 adjusted | Parity-1 adjusted |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in mechanisms:
        p0 = parity["0"][name]
        p1 = parity["1"][name]
        lines.append(
            f"| `{name}` | {sci(p0['combo_final_mse']['mean'])} | "
            f"{sci(p1['combo_final_mse']['mean'])} | "
            f"{sci(p0['combo_adjusted_score']['mean'])} | "
            f"{sci(p1['combo_adjusted_score']['mean'])} |"
        )

    lines.extend([
        "",
        "## Mean Classification Counts",
        "",
        "The table shows the mean number of neurons omitted as dead. N/P intentionally classify "
        "no neuron as always-active on layers 17-30; those neurons stay in the sampled ReLU path. "
        "Layers 31-32 retain the established symmetric classifier.",
        "",
        "| Layer | M dead | M active | N dead | P dead | Q dead |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    counts = {name: mean_class_counts(metadata, name) for name in mechanisms}
    for layer in range(17, 33):
        lines.append(
            f"| {layer} | {counts[BASELINE][layer]['dead']:.1f} | "
            f"{counts[BASELINE][layer]['active']:.1f} | "
            f"{counts['N_last16_dead_only_same'][layer]['dead']:.1f} | "
            f"{counts['P_last16_dead_only_m050'][layer]['dead']:.1f} | "
            f"{counts['Q_last16_early_dead_m075'][layer]['dead']:.1f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "1. **Always-active linearization was not helping accuracy.** N uses the same dead "
        "thresholds as M but leaves every non-dead neuron on layers 17-30 in the sampled ReLU "
        "path. Its structural-versus-dense MSE is lower, although the raw-MSE direction differs "
        "between parity folds and it does not save counted FLOPs.",
        "2. **Q is the robust conservative efficiency point.** It only makes layers 17-20 more "
        "aggressive. Raw error and structural error are effectively unchanged while counted "
        "FLOPs fall by about one percent.",
        "3. **P is the aggressive score point.** It saves about three percent counted FLOPs and "
        "wins adjusted score in both parity folds, but structural error rises substantially and "
        "the raw-MSE direction is not parity-stable. It is useful as an efficiency option, not an "
        "accuracy improvement.",
        "4. **The dead decision is analytic, not sample-fitted.** Corrected K2 propagates mean and "
        "covariance from the known Gaussian input through the submitted weights. A neuron is "
        "declared dead when its standardized preactivation `alpha = mu / sigma` is at or below "
        "the layer threshold with a negative sign. The Sobol paths are evaluated only after this "
        "classification; dense sample signs are used solely to audit the approximation offline.",
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
