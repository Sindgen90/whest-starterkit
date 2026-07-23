"""Estimate small active-subspace spectra before implementing the active estimator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.src.active_subspaces.gradients import input_jacobian_transpose_probe
from research.src.rqmc.points import LatticeConfig, numpy_lattice_gaussians

RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"


def selected_rows(partition: str, limit: int | None) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if "splits" in manifest:
        ids = set(manifest["splits"][partition]["mlp_ids"])
    else:
        ids = set(manifest[f"{partition}_mlp_ids"])
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    selected = sorted((row for row in rows if int(row["mlp_id"]) in ids), key=lambda row: int(row["mlp_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def estimate_spectrum(row: dict, pilot_samples: int, probes_per_sample: int) -> dict:
    weights = [np.asarray(layer, dtype=np.float64) for layer in row["weights"]]
    width = int(weights[0].shape[0])
    x_pilot = numpy_lattice_gaussians(
        LatticeConfig(n_samples=pilot_samples, seed_offset=31_415_927, multiplier=17_797),
        width=width,
        seed=int(row["mlp_seed"]),
    )
    rng = np.random.default_rng(int(row["mlp_seed"]) + 27_182_818)
    gradients = []
    probe_scale = 1.0 / np.sqrt(width)
    for x in x_pilot:
        for _ in range(probes_per_sample):
            probe = rng.choice(np.array([-probe_scale, probe_scale]), size=width)
            gradients.append(input_jacobian_transpose_probe(weights, x, probe))
    g = np.asarray(gradients, dtype=np.float64).T
    covariance = (g @ g.T) / float(g.shape[1])
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    total = float(np.sum(np.maximum(eigenvalues, 0.0)))
    if total > 0.0:
        cumulative = np.cumsum(np.maximum(eigenvalues, 0.0)) / total
    else:
        cumulative = np.zeros_like(eigenvalues)
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "width": width,
        "pilot_samples": pilot_samples,
        "probes_per_sample": probes_per_sample,
        "gradient_columns": int(g.shape[1]),
        "top_eigenvalues": [float(x) for x in eigenvalues[:12]],
        "energy_rank_4": float(cumulative[3]) if len(cumulative) >= 4 else float(cumulative[-1]),
        "energy_rank_8": float(cumulative[7]) if len(cumulative) >= 8 else float(cumulative[-1]),
        "energy_rank_16": float(cumulative[15]) if len(cumulative) >= 16 else float(cumulative[-1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation"], default="development")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--pilot-samples", type=int, default=8)
    parser.add_argument("--probes-per-sample", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(args.partition, args.limit)
    records = [estimate_spectrum(row, args.pilot_samples, args.probes_per_sample) for row in rows]
    jsonl_path = RESULTS / "active_spectrum_diagnostics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report_path = REPORTS / "active_spectrum_diagnostics.md"
    table_rows = [
        "| MLP id | grad cols | E4 | E8 | E16 | top eigenvalues |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        top = ", ".join(f"{value:.3g}" for value in record["top_eigenvalues"][:5])
        table_rows.append(
            f"| {record['mlp_id']} | {record['gradient_columns']} | "
            f"{record['energy_rank_4']:.3f} | {record['energy_rank_8']:.3f} | "
            f"{record['energy_rank_16']:.3f} | {top} |"
        )
    report_path.write_text(
        "# Active-spectrum diagnostics\n\n"
        "This is a pre-estimator diagnostic only. It estimates leading eigenspace concentration "
        "from manual ReLU reverse-mode input gradients on the frozen development partition.\n\n"
        + "\n".join(table_rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {jsonl_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
