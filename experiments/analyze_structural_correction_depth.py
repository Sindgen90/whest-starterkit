"""Analyze correction/depth ablations and merge them with the structural ladder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "structural_correction_depth_limit50.json"
DEFAULT_CACHE = ROOT / "results" / "structural_correction_depth_limit50.npz"
DEFAULT_LADDER = ROOT / "results" / "deep_structural_tail_last8_limit50.json"
DEFAULT_REPORT = ROOT / "research" / "reports" / "structural_correction_depth.md"
BASELINE = "F_last8_control"


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
    metadata: dict[str, Any],
    ladder: dict[str, Any],
    paired: dict[str, dict[str, float]],
    output: Path,
) -> None:
    summary = metadata["summary"]["crossfit_mean"]
    ladder_summary = ladder["summary"]["crossfit_mean"]
    parity = metadata["summary"]["parity_crossfit"]
    lines = [
        "# Structural Corrections and Tail Depth",
        "",
        "All ablations use the exact 32,768-point antithetic Sobol asset and corrected K2. "
        "They were evaluated in balanced order on the first 50 Mini MLPs with opposite-parity "
        "lambda fitting. No mechanisms are combined in this experiment.",
        "",
        "## Existing Depth Ladder",
        "",
        "| Candidate | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "current_last2_t3", "C_last6_tapered", "D_last8_conservative",
        "E_last8_balanced", "F_last8_aggressive",
    ):
        row = ladder_summary[name]
        lines.append(
            f"| `{name}` | {sci(row['combo_final_mse'])} | {sci(row['flops'])} | "
            f"{sci(row['combo_adjusted_score'])} | {sci(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## New Isolated Ablations",
        "",
        "| Candidate | Mechanism | Raw final MSE | FLOPs | Full adjusted | Flop-only adjusted | Structural vs dense |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    mechanisms = {
        "F_last8_control": "last-eight control",
        "G_last8_dead_tail": "dead Gaussian tail mean",
        "H_last8_amplification": "multistep amplification thresholds",
        "I_last8_tail_risk": "tail probability x outgoing norm",
        "J_last10_tapered": "last ten",
        "K_last12_tapered": "last twelve",
    }
    for name, mechanism in mechanisms.items():
        row = summary[name]
        lines.append(
            f"| `{name}` | {mechanism} | {sci(row['combo_final_mse'])} | "
            f"{sci(row['flops'])} | {sci(row['combo_adjusted_score'])} | "
            f"{sci(row['combo_flop_only_adjusted_score'])} | "
            f"{sci(row['structural_vs_dense_final_mse'])} |"
        )

    lines.extend([
        "",
        "## Paired Against F",
        "",
        "Intervals are paired 20,000-draw bootstrap intervals for candidate minus F.",
        "",
        "| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in list(mechanisms)[1:]:
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
    for name in mechanisms:
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
        "## Findings",
        "",
        "1. **Dead-tail correction does not hold up.** It slightly improves aggregate raw MSE, "
        "but the parity directions disagree, structural-versus-dense error rises, and carrying the "
        "constants increases residual compute enough to lose adjusted score.",
        "2. **Multistep amplification is neutral in this form.** It makes a tiny raw improvement but "
        "does not improve full adjusted score.",
        "3. **Tail-risk reallocation is the useful accuracy signal.** At effectively identical "
        "counted FLOPs it lowers raw MSE and reduces structural-versus-dense final MSE from "
        "`3.3344e-10` to `2.9585e-10`. Its current NumPy controller overhead makes full adjusted "
        "score slightly worse; this mechanism is worth retaining for a later optimized combination.",
        "4. **Greater depth continues to pay.** J and K preserve raw accuracy while reducing FLOPs. "
        "K is the strongest standalone result and wins adjusted score in both parity directions.",
        "",
        "K uses thresholds `4.75, 4.5, 4.25, 4.0, 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, "
        "3.0, 3.0` on layers 21-32. Its empirical wrong-sign fractions at layers 21-24 are "
        "approximately `2.6e-8`, `6e-9`, `1.5e-7`, and `7.9e-7`; the newly introduced "
        "per-layer structural MSE remains below `2.2e-13` through layer 24.",
        "",
        "No correction or adaptive rule is combined with J/K here. That remains a separate next "
        "experiment, as requested.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.json.read_text(encoding="utf-8"))
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    with np.load(args.cache) as cache:
        paired = paired_summary(metadata, cache, args.bootstrap_draws)
    write_report(metadata, ladder, paired, args.output)
    print(json.dumps(paired, indent=2, sort_keys=True))
    print(f"wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
