"""Diagnostic for cumulant-diagonalizing activation bases.

This implements the evidence test from the basis-change idea:

    R3(Q) = sum_i kappa_iii(Q^T X)^2 / ||kappa_3(X)||_F^2
    R4(Q) = sum_i kappa_iiii(Q^T X)^2 / ||kappa_4(X)||_F^2

The denominators are estimated from offline activation samples without forming
dense third/fourth-order tensors.  The third moment denominator uses the Gram
identity ||E[X^⊗3]||_F^2 = E[(X.X')^3].  The fourth cumulant denominator uses
the empirical fourth central moment minus Gaussian pairings.
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
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
LAYERS = (4, 8, 16, 24, 32)
BASES = (
    "original_neuron",
    "random_orthogonal",
    "pca_rotation",
    "whitened_pca",
    "whitened_random",
    "whitened_skew_power",
)


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


def propagate_and_collect(row: dict[str, Any], samples: int, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
    weights = np.asarray(row["weights"], dtype=np.float64)
    rng = np.random.default_rng(int(row["mlp_seed"]) + 41_000_033)
    x = rng.standard_normal(size=(samples, weights.shape[1])).astype(np.float64)
    wanted = set(layers)
    activations: dict[int, np.ndarray] = {}
    for layer_index, weight in enumerate(weights, start=1):
        x = np.maximum(x @ weight, 0.0)
        if layer_index in wanted:
            activations[layer_index] = x.copy()
    return activations


def random_orthogonal(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal(size=(width, width)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def pca_whiten(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = centered.T @ centered / centered.shape[0]
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], max(float(values.max()), 1.0) * 1e-10)
    vectors = vectors[:, order]
    rotated = centered @ vectors
    whitened = rotated / np.sqrt(values)[None, :]
    return rotated, whitened, values


def orthogonalize(v: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    for previous in columns:
        v = v - previous * float(previous @ v)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-300:
        return v
    return v / norm


def skew_power_basis(z: np.ndarray, rank: int, iterations: int, seed: int) -> np.ndarray:
    width = z.shape[1]
    rng = np.random.default_rng(seed)
    columns: list[np.ndarray] = []
    for _ in range(min(rank, width)):
        v = rng.standard_normal(width)
        v = orthogonalize(v, columns)
        if np.linalg.norm(v) <= 1e-300:
            break
        for _ in range(iterations):
            projection = z @ v
            update = z.T @ (projection * projection) / z.shape[0]
            v = orthogonalize(update, columns)
            if np.linalg.norm(v) <= 1e-300:
                v = orthogonalize(rng.standard_normal(width), columns)
        if np.linalg.norm(v) > 1e-300:
            columns.append(v)
    completion = rng.standard_normal(size=(width, width))
    if columns:
        seeded = np.column_stack([*columns, completion])
    else:
        seeded = completion
    q, _ = np.linalg.qr(seeded)
    return q[:, :width]


def third_frobenius_sq(centered: np.ndarray) -> float:
    gram = centered @ centered.T
    return float(np.mean(gram * gram * gram))


def fourth_cumulant_frobenius_sq(centered: np.ndarray, cov: np.ndarray) -> float:
    gram = centered @ centered.T
    m4_norm = float(np.mean(gram ** 4))
    sigma_x = centered @ cov
    quadratic = np.sum(sigma_x * centered, axis=1)
    moment_pairing_inner = float(np.mean(quadratic * quadratic))
    cov_fro_sq = float(np.sum(cov * cov))
    cov_values = np.linalg.eigvalsh(cov)
    trace_cov4 = float(np.sum(cov_values ** 4))
    value = m4_norm - 6.0 * moment_pairing_inner + 3.0 * cov_fro_sq * cov_fro_sq + 6.0 * trace_cov4
    return max(value, 0.0)


def cumulant_diag_ratios(y: np.ndarray) -> dict[str, float]:
    centered = y - y.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / centered.shape[0]
    diag_var = np.diag(cov)
    diag_k3 = np.mean(centered ** 3, axis=0)
    diag_k4 = np.mean(centered ** 4, axis=0) - 3.0 * diag_var * diag_var
    denom3 = third_frobenius_sq(centered)
    denom4 = fourth_cumulant_frobenius_sq(centered, cov)
    return {
        "r3_diag_fraction": float(np.sum(diag_k3 * diag_k3) / denom3) if denom3 > 0.0 else float("nan"),
        "r4_diag_fraction": float(np.sum(diag_k4 * diag_k4) / denom4) if denom4 > 0.0 else float("nan"),
        "third_frobenius_sq": denom3,
        "fourth_cumulant_frobenius_sq": denom4,
        "mean_abs_diag_skew": float(np.mean(np.abs(diag_k3))),
        "mean_abs_diag_kurtosis_cumulant": float(np.mean(np.abs(diag_k4))),
    }


def evaluate_layer(centered: np.ndarray, row_seed: int, layer: int, args: argparse.Namespace) -> list[dict[str, float | str]]:
    width = centered.shape[1]
    pca_rotated, whitened, eigenvalues = pca_whiten(centered)
    bases: dict[str, np.ndarray] = {
        "original_neuron": centered,
        "random_orthogonal": centered @ random_orthogonal(width, row_seed + layer + 101),
        "pca_rotation": pca_rotated,
        "whitened_pca": whitened,
        "whitened_random": whitened @ random_orthogonal(width, row_seed + layer + 202),
        "whitened_skew_power": whitened
        @ skew_power_basis(
            whitened,
            rank=args.skew_rank,
            iterations=args.skew_iterations,
            seed=row_seed + layer + 303,
        ),
    }
    condition_number = float(eigenvalues[0] / max(eigenvalues[-1], 1e-300))
    records = []
    for basis_name, values in bases.items():
        ratios = cumulant_diag_ratios(values)
        ratios["basis"] = basis_name
        ratios["cov_condition_number"] = condition_number
        records.append(ratios)
    return records


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    start = time.perf_counter()
    activations = propagate_and_collect(row, args.samples, tuple(args.layers))
    runtime_s = time.perf_counter() - start
    records = []
    for layer, values in activations.items():
        centered = values - values.mean(axis=0, keepdims=True)
        for basis_record in evaluate_layer(centered, int(row["mlp_seed"]), layer, args):
            record = {
                "schema_version": "1.0",
                "run_id": args.run_id,
                "created_at_utc": args.created_at,
                "partition": args.partition,
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "layer": layer,
                "samples": args.samples,
                "skew_rank": args.skew_rank,
                "skew_iterations": args.skew_iterations,
                "runtime_s_for_mlp": runtime_s,
            }
            record.update(basis_record)
            records.append(record)
    return records


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for layer in args.layers:
        for basis in BASES:
            selected = [
                record
                for record in records
                if int(record["layer"]) == int(layer) and record["basis"] == basis
            ]
            if not selected:
                continue
            rows.append(
                {
                    "partition": args.partition,
                    "layer": layer,
                    "basis": basis,
                    "n_mlps": len(selected),
                    "samples": args.samples,
                    "skew_rank": args.skew_rank,
                    "skew_iterations": args.skew_iterations,
                    "mean_r3_diag_fraction": mean_or_nan([record["r3_diag_fraction"] for record in selected]),
                    "median_r3_diag_fraction": float(
                        np.nanmedian([record["r3_diag_fraction"] for record in selected])
                    ),
                    "mean_r4_diag_fraction": mean_or_nan([record["r4_diag_fraction"] for record in selected]),
                    "median_r4_diag_fraction": float(
                        np.nanmedian([record["r4_diag_fraction"] for record in selected])
                    ),
                    "mean_third_frobenius_sq": mean_or_nan([record["third_frobenius_sq"] for record in selected]),
                    "mean_fourth_cumulant_frobenius_sq": mean_or_nan(
                        [record["fourth_cumulant_frobenius_sq"] for record in selected]
                    ),
                    "mean_cov_condition_number": mean_or_nan(
                        [record["cov_condition_number"] for record in selected]
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cumulant-diagonalization diagnostic",
        "",
        "This tests whether an activation basis change concentrates third/fourth cumulant energy onto marginal diagonal cumulants.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Mean diagonal cumulant fractions",
        "",
        "| layer | basis | R3 diag fraction | R4 diag fraction |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {basis} | {r3:.3f} | {r4:.3f} |".format(
                layer=row["layer"],
                basis=row["basis"],
                r3=row["mean_r3_diag_fraction"],
                r4=row["mean_r4_diag_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "A diagonal-cumulant estimator deserves implementation only if a basis consistently pushes R3/R4 diagonal fractions into roughly the 0.5-0.8 range.  If whitening/skew-power does not materially improve over raw coordinates, stop this branch.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="development")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--skew-rank", type=int, default=16)
    parser.add_argument("--skew-iterations", type=int, default=8)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.skew_rank <= 0:
        raise SystemExit("--skew-rank must be positive")
    if args.skew_iterations <= 0:
        raise SystemExit("--skew-iterations must be positive")
    args.layers = tuple(sorted(set(args.layers)))
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in __import__("sys").argv]
    rows = select_rows(args.partition, args.limit)
    records: list[dict[str, Any]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "cumulant_diagonalization_runs.jsonl"
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
    csv_path = RESULTS / f"cumulant_diagonalization_{suffix}_summary.csv"
    report_path = REPORTS / f"cumulant_diagonalization_{suffix}.md"
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
