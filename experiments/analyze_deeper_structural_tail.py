"""Analyze last-14 and last-16 candidates against last-12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "structural_deeper_tail_limit50.json"
DEFAULT_CACHE = ROOT / "results" / "structural_deeper_tail_limit50.npz"
DEFAULT_REPORT = ROOT / "research" / "reports" / "deeper_structural_tail.md"
BASELINE = "K_last12_control"


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
        row = {}
        for metric in ("raw", "adjusted", "flop_adjusted", "flops", "effective"):
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


def sci(value: float) -> str:
    return f"{value:.4e}"


def write_report(
    metadata: dict[str, Any], paired: dict[str, dict[str, float]], output: Path
) -> None:
    summary = metadata["summary"]["crossfit_mean"]
    parity = metadata["summary"]["parity_crossfit"]
    candidates = metadata["configuration"]["candidates"]
    lines = [
        "# Deeper Structural Tail Scout",
        "",
        "Last fourteen and last sixteen were tested in isolation on the balanced first 50 Mini "
        "MLPs. The established F-last-eight and K-last-twelve estimators were rerun as exact controls.",
        "",
        "| Candidate | Thresholds | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        name = candidate["name"]
        row = summary[name]
        thresholds = ", ".join(f"{value:g}" for value in candidate["thresholds"])
        lines.append(
            f"| `{name}` | {thresholds} | {sci(row['combo_final_mse'])} | "
            f"{sci(row['flops'])} | {sci(row['combo_adjusted_score'])} | "
            f"{sci(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## Paired Against K",
        "",
        "| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("L_last14_tapered", "M_last16_tapered"):
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
    for name in (BASELINE, "L_last14_tapered", "M_last16_tapered"):
        p0 = parity["0"][name]
        p1 = parity["1"][name]
        lines.append(
            f"| `{name}` | {sci(p0['combo_final_mse']['mean'])} | "
            f"{sci(p0['combo_adjusted_score']['mean'])} | "
            f"{sci(p1['combo_final_mse']['mean'])} | "
            f"{sci(p1['combo_adjusted_score']['mean'])} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Both L and M preserve raw accuracy and improve adjusted score in both parity directions. "
        "M is the stronger FLOP-only result, while its measured effective compute is nearly tied "
        "with L because the additional narrow operations approach the local dispatch-overhead floor.",
        "",
        "M classifies only about 11 neurons at layer 17 and 18 neurons at layer 18; the remaining "
        "neurons are still sampled with ReLU. L classifies about 25 neurons at layer 19 and 33 at "
        "layer 20. Across all 50 MLPs, dense 32k diagnostics observed no wrong-sign samples for "
        "the neurons classified at layers 17-20.",
        "",
        "The result is positive but clearly in diminishing-return territory. M is the standalone "
        "winner; extending still earlier should be considered only with a threshold/controller "
        "study rather than blindly continuing the same taper.",
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
