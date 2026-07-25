"""Estimate third-cumulant mode rank from offline activation samples.

For centered activations X at a selected layer, the third cumulant is the third
central moment tensor T.  We probe its mode space through contractions:

    V[:, m] = T[g_m, g_m, :]
            = E[(X @ g_m)^2 X].

For the "generic" all-distinct-index residual, we subtract contractions whose
tensor entries include repeated indices.  This approximates the question behind
the Tucker-K3 proposal: after preserving diagonal/repeated-index corrections,
does the remaining generic third-cumulant tensor have a low-dimensional mode
space?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
RANKS = (4, 8, 16, 32, 64)
LAYERS = (4, 8, 16, 24, 32)
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"


def select_rows(partition: str, limit: int | None) -> list[dict[str, Any]]:
    if partition == "holdout":
        raise SystemExit("Holdout is gated for cumulant diagnostics.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = set(int(x) for x in manifest[f"{partition}_mlp_ids"])
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    selected = sorted(
        [row for row in rows if int(row["mlp_id"]) in ids],
        key=lambda row: int(row["mlp_id"]),
    )
    if limit is not None:
        selected = selected[:limit]
    return selected


def normal_samples(n_samples: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_samples, width)).astype(np.float64)


def normalized_probes(count: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal(size=(count, width)).astype(np.float64)
    norms = np.linalg.norm(probes, axis=1, keepdims=True)
    return probes / np.maximum(norms, 1e-300)


def propagate_and_collect(row: dict[str, Any], samples: int, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
    weights = np.asarray(row["weights"], dtype=np.float64)
    x = normal_samples(samples, weights.shape[1], int(row["mlp_seed"]) + 31_415_927)
    wanted = set(layers)
    activations: dict[int, np.ndarray] = {}
    for layer_index, weight in enumerate(weights, start=1):
        x = np.maximum(x @ weight, 0.0)
        if layer_index in wanted:
            activations[layer_index] = x.copy()
    return activations


def contraction_matrices(centered: np.ndarray, probes: np.ndarray) -> dict[str, np.ndarray]:
    n_samples = centered.shape[0]
    projections = centered @ probes.T
    full = centered.T @ (projections * projections) / n_samples

    # R[out, squared_coord] = E[X_out * X_squared_coord^2].
    repeated = centered.T @ (centered * centered) / n_samples
    diagonal = np.diag(repeated)

    first_repeated = repeated @ (probes * probes).T
    inner = repeated.T @ probes.T
    second_repeated = 2.0 * probes.T * (inner - diagonal[:, None] * probes.T)
    repeated_contribution = first_repeated + second_repeated
    generic = full - repeated_contribution
    return {
        "full": full,
        "protected_repeated": repeated_contribution,
        "generic_all_distinct": generic,
    }


def spectrum_metrics(matrix: np.ndarray, ranks: tuple[int, ...]) -> dict[str, float]:
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = singular_values * singular_values
    total = float(energy.sum())
    result = {
        "energy": total,
        "effective_rank": float((energy.sum() ** 2) / np.maximum(np.sum(energy * energy), 1e-300)),
    }
    cumulative = np.cumsum(energy)
    for rank in ranks:
        capped = min(rank, len(cumulative))
        result[f"energy_rank_{rank}"] = float(cumulative[capped - 1] / total) if total > 0.0 else 0.0
    return result


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    started = time.perf_counter()
    activations = propagate_and_collect(row, args.samples, tuple(args.layers))
    runtime_s = time.perf_counter() - started
    records = []
    for layer, values in activations.items():
        centered = values - values.mean(axis=0, keepdims=True)
        probes = normalized_probes(args.probes, centered.shape[1], int(row["mlp_seed"]) + 2_718_281 + layer)
        matrices = contraction_matrices(centered, probes)
        full_energy = float(np.sum(matrices["full"] * matrices["full"]))
        generic_energy = float(np.sum(matrices["generic_all_distinct"] * matrices["generic_all_distinct"]))
        protected_energy = float(np.sum(matrices["protected_repeated"] * matrices["protected_repeated"]))
        for tensor_part, matrix in matrices.items():
            metrics = spectrum_metrics(matrix, tuple(args.ranks))
            record = {
                "schema_version": "1.0",
                "run_id": args.run_id,
                "created_at_utc": args.created_at,
                "partition": args.partition,
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "layer": layer,
                "tensor_part": tensor_part,
                "samples": args.samples,
                "probes": args.probes,
                "runtime_s_for_mlp": runtime_s,
                "full_energy": full_energy,
                "generic_energy": generic_energy,
                "protected_repeated_energy": protected_energy,
                "generic_energy_fraction_of_full": generic_energy / full_energy if full_energy else float("nan"),
                "protected_energy_fraction_of_full": protected_energy / full_energy if full_energy else float("nan"),
            }
            record.update(metrics)
            records.append(record)
    return records


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr))


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for tensor_part in ("full", "protected_repeated", "generic_all_distinct"):
        for layer in args.layers:
            selected = [
                record
                for record in records
                if record["tensor_part"] == tensor_part and int(record["layer"]) == int(layer)
            ]
            if not selected:
                continue
            row = {
                "partition": args.partition,
                "tensor_part": tensor_part,
                "layer": layer,
                "n_mlps": len(selected),
                "samples": args.samples,
                "probes": args.probes,
                "mean_effective_rank": mean_or_nan([record["effective_rank"] for record in selected]),
                "mean_energy": mean_or_nan([record["energy"] for record in selected]),
                "mean_generic_energy_fraction_of_full": mean_or_nan(
                    [record["generic_energy_fraction_of_full"] for record in selected]
                ),
            }
            for rank in args.ranks:
                key = f"energy_rank_{rank}"
                row[f"mean_{key}"] = mean_or_nan([record[key] for record in selected])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    generic_rows = [row for row in summary_rows if row["tensor_part"] == "generic_all_distinct"]
    rank_columns = [rank for rank in (4, 8, 16, 32, 64) if rank in args.ranks]
    lines = [
        "# Cumulant mode-rank diagnostic",
        "",
        "This tests the Tucker-K3 prerequisite: after removing diagonal/repeated-index protected structure, do third-cumulant contractions live in a low-dimensional mode space?",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Generic all-distinct contraction spectrum",
        "",
        "| layer | "
        + " | ".join(f"rank-{rank}" for rank in rank_columns)
        + " | effective rank | generic/full energy |",
        "| ---: | " + " | ".join("---:" for _ in rank_columns) + " | ---: | ---: |",
    ]
    for row in generic_rows:
        rank_values = " | ".join(f"{row[f'mean_energy_rank_{rank}']:.3f}" for rank in rank_columns)
        lines.append(
            f"| {row['layer']} | {rank_values} | "
            f"{row['mean_effective_rank']:.2f} | "
            f"{row['mean_generic_energy_fraction_of_full']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Tucker-K3 is promising only if rank 16 or 32 captures most generic contraction energy consistently.  If rank-32 is weak or the generic residual is tiny/noisy, do not implement a full Tucker estimator in Phase 1.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="development")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--probes", type=int, default=96)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.probes <= 0:
        raise SystemExit("--probes must be positive")
    args.layers = tuple(sorted(set(args.layers)))
    args.ranks = tuple(sorted(set(args.ranks)))
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in __import__("sys").argv]
    rows = select_rows(args.partition, args.limit)
    records: list[dict[str, Any]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "cumulant_mode_rank_runs.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            row_records = evaluate_row(row, args)
            records.extend(row_records)
            for record in row_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"{index}/{len(rows)} mlp_id={row['mlp_id']} completed", flush=True)
    summary_rows = summarize(records, args)
    suffix = args.partition if args.limit is None else f"{args.partition}_limit{args.limit}"
    csv_path = RESULTS / f"cumulant_mode_rank_{suffix}_summary.csv"
    report_path = REPORTS / f"cumulant_mode_rank_{suffix}.md"
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, summary_rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "jsonl": str(jsonl_path),
                "summary_csv": str(csv_path),
                "report": str(report_path),
                "summary": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
