"""Oracle one-dimensional ReLU activation-update diagnostic.

For sampled preactivations Z at selected layers/neurons, estimate near-oracle
univariate cumulants and compare formulas for E[ReLU(Z)]:

* Gaussian moment closure;
* Edgeworth K3 correction;
* Edgeworth K3+K4 correction;
* Lugannani-Rice saddlepoint tail integration from a truncated K4 CGF.

This isolates activation-update error from cumulant-propagation error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
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
METHODS = ("gaussian", "edgeworth_k3", "edgeworth_k34", "saddlepoint_k34")


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def select_rows(partition: str, limit: int | None) -> list[dict[str, Any]]:
    if partition == "holdout":
        raise SystemExit("Holdout is gated for activation-update diagnostics.")
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


def selected_neurons(width: int, count: int) -> np.ndarray:
    if count >= width:
        return np.arange(width, dtype=np.int64)
    return np.unique(np.linspace(0, width - 1, count, dtype=np.int64))


def relu_gaussian(mu: float, var: float) -> float:
    sigma = math.sqrt(max(var, 1e-30))
    alpha = mu / sigma
    return mu * normal_cdf(alpha) + sigma * normal_pdf(alpha)


def edgeworth_relu(mu: float, var: float, k3: float, k4: float, include_k4: bool) -> float:
    sigma = math.sqrt(max(var, 1e-30))
    alpha = mu / sigma
    phi = normal_pdf(alpha)
    base = mu * normal_cdf(alpha) + sigma * phi
    third_derivative = -alpha * phi / max(var, 1e-30)
    estimate = base + k3 * third_derivative / 6.0
    if include_k4:
        fourth_derivative = (alpha * alpha - 1.0) * phi / max(sigma ** 3, 1e-30)
        estimate += k4 * fourth_derivative / 24.0
    return estimate


def cgf(mu: float, var: float, k3: float, k4: float, t: float) -> float:
    return mu * t + 0.5 * var * t * t + k3 * t ** 3 / 6.0 + k4 * t ** 4 / 24.0


def cgf_prime(mu: float, var: float, k3: float, k4: float, t: float) -> float:
    return mu + var * t + 0.5 * k3 * t * t + k4 * t ** 3 / 6.0


def cgf_second(var: float, k3: float, k4: float, t: float) -> float:
    return var + k3 * t + 0.5 * k4 * t * t


def saddlepoint_root(mu: float, var: float, k3: float, k4: float, x: float) -> float | None:
    t = (x - mu) / max(var, 1e-30)
    for _ in range(30):
        second = cgf_second(var, k3, k4, t)
        if not math.isfinite(second) or second <= 1e-14:
            return None
        step = (cgf_prime(mu, var, k3, k4, t) - x) / second
        if not math.isfinite(step):
            return None
        t -= max(-2.0, min(2.0, step))
        if abs(step) <= 1e-10 * (1.0 + abs(t)):
            return t
    return t if math.isfinite(t) and cgf_second(var, k3, k4, t) > 1e-14 else None


def saddlepoint_cdf(mu: float, var: float, k3: float, k4: float, x: float) -> float | None:
    sigma = math.sqrt(max(var, 1e-30))
    if abs(x - mu) < 1e-7 * sigma:
        return 0.5
    t = saddlepoint_root(mu, var, k3, k4, x)
    if t is None or abs(t) < 1e-14:
        return None
    k_value = cgf(mu, var, k3, k4, t)
    saddle = t * x - k_value
    if not math.isfinite(saddle) or saddle <= 0.0:
        return None
    second = cgf_second(var, k3, k4, t)
    if second <= 0.0 or not math.isfinite(second):
        return None
    w = math.copysign(math.sqrt(2.0 * saddle), t)
    u = t * math.sqrt(second)
    if abs(w) < 1e-12 or abs(u) < 1e-12:
        return None
    cdf = normal_cdf(w) + normal_pdf(w) * (1.0 / w - 1.0 / u)
    if not math.isfinite(cdf):
        return None
    return min(1.0, max(0.0, cdf))


def saddlepoint_relu(mu: float, var: float, k3: float, k4: float, nodes: int) -> tuple[float | None, str]:
    sigma = math.sqrt(max(var, 1e-30))
    upper = max(mu + 10.0 * sigma, 10.0 * sigma)
    if upper <= 0.0 or not math.isfinite(upper):
        return None, "bad_upper"
    xs, ws = np.polynomial.legendre.leggauss(nodes)
    grid = 0.5 * upper * (xs + 1.0)
    weights = 0.5 * upper * ws
    total = 0.0
    for x, weight in zip(grid, weights):
        cdf = saddlepoint_cdf(mu, var, k3, k4, float(x))
        if cdf is None:
            return None, "root_or_cdf_failed"
        total += float(weight) * max(0.0, 1.0 - cdf)
    return max(0.0, total), "completed"


def cumulants(z: np.ndarray) -> dict[str, float]:
    mu = float(np.mean(z))
    centered = z - mu
    var = float(np.mean(centered * centered))
    third = float(np.mean(centered ** 3))
    fourth_cumulant = float(np.mean(centered ** 4) - 3.0 * var * var)
    truth = float(np.mean(np.maximum(z, 0.0)))
    return {"mu": mu, "var": var, "k3": third, "k4": fourth_cumulant, "truth": truth}


def evaluate_method(stats: dict[str, float], method: str, nodes: int) -> tuple[float | None, str]:
    mu, var, k3, k4 = stats["mu"], stats["var"], stats["k3"], stats["k4"]
    try:
        if method == "gaussian":
            return relu_gaussian(mu, var), "completed"
        if method == "edgeworth_k3":
            return edgeworth_relu(mu, var, k3, k4, include_k4=False), "completed"
        if method == "edgeworth_k34":
            return edgeworth_relu(mu, var, k3, k4, include_k4=True), "completed"
        if method == "saddlepoint_k34":
            return saddlepoint_relu(mu, var, k3, k4, nodes)
    except Exception as exc:  # pragma: no cover - diagnostic resilience
        return None, f"{type(exc).__name__}: {exc}"
    raise ValueError(f"Unknown method: {method}")


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    weights = np.asarray(row["weights"], dtype=np.float64)
    width = int(weights.shape[1])
    neurons = selected_neurons(width, args.neurons)
    rng = np.random.default_rng(int(row["mlp_seed"]) + 59_000_021)
    x = rng.standard_normal(size=(args.samples, width)).astype(np.float64)
    records = []
    started = time.perf_counter()
    wanted_layers = set(args.layers)
    for layer, weight in enumerate(weights, start=1):
        z = x @ weight
        if layer in wanted_layers:
            for neuron in neurons:
                stats = cumulants(z[:, neuron])
                for method in METHODS:
                    estimate, status = evaluate_method(stats, method, args.saddlepoint_nodes)
                    abs_error = float("nan") if estimate is None else abs(estimate - stats["truth"])
                    squared_error = float("nan") if estimate is None else (estimate - stats["truth"]) ** 2
                    records.append(
                        {
                            "schema_version": "1.0",
                            "run_id": args.run_id,
                            "created_at_utc": args.created_at,
                            "partition": args.partition,
                            "mlp_id": int(row["mlp_id"]),
                            "mlp_name": str(row["mlp_name"]),
                            "mlp_seed": int(row["mlp_seed"]),
                            "layer": layer,
                            "neuron": int(neuron),
                            "method": method,
                            "status": status,
                            "samples": args.samples,
                            "truth": stats["truth"],
                            "estimate": estimate,
                            "abs_error": abs_error,
                            "squared_error": squared_error,
                            "mu": stats["mu"],
                            "var": stats["var"],
                            "k3": stats["k3"],
                            "k4": stats["k4"],
                        }
                    )
        x = np.maximum(z, 0.0)
    runtime_s = time.perf_counter() - started
    for record in records:
        record["runtime_s_for_mlp"] = runtime_s
    return records


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"mean": float("nan"), "median": float("nan"), "standard_error": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "standard_error": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "max": float(arr.max()),
    }


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    completed = [record for record in records if record["status"] == "completed"]
    gaussian_by_key = {
        (record["mlp_id"], record["layer"], record["neuron"]): record
        for record in completed
        if record["method"] == "gaussian"
    }
    for method in METHODS:
        for layer in args.layers:
            selected = [record for record in records if record["method"] == method and int(record["layer"]) == int(layer)]
            done = [record for record in selected if record["status"] == "completed"]
            abs_stats = aggregate([record["abs_error"] for record in done])
            sq_stats = aggregate([record["squared_error"] for record in done])
            paired = [
                record
                for record in done
                if method != "gaussian" and (record["mlp_id"], record["layer"], record["neuron"]) in gaussian_by_key
            ]
            if paired:
                wins = [
                    record["squared_error"]
                    < gaussian_by_key[(record["mlp_id"], record["layer"], record["neuron"])]["squared_error"]
                    for record in paired
                ]
                mean_delta = float(
                    np.mean(
                        [
                            record["squared_error"]
                            - gaussian_by_key[(record["mlp_id"], record["layer"], record["neuron"])]["squared_error"]
                            for record in paired
                        ]
                    )
                )
            else:
                wins = []
                mean_delta = float("nan")
            rows.append(
                {
                    "partition": args.partition,
                    "layer": layer,
                    "method": method,
                    "n_requested": len(selected),
                    "n_completed": len(done),
                    "failure_fraction": 1.0 - len(done) / len(selected) if selected else float("nan"),
                    "mean_abs_error": abs_stats["mean"],
                    "median_abs_error": abs_stats["median"],
                    "max_abs_error": abs_stats["max"],
                    "mean_squared_error": sq_stats["mean"],
                    "median_squared_error": sq_stats["median"],
                    "win_fraction_vs_gaussian": float(np.mean(wins)) if wins else float("nan"),
                    "mean_squared_error_delta_vs_gaussian": mean_delta,
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
        "# Oracle activation-update diagnostic",
        "",
        "This isolates the scalar ReLU update error using sampled preactivation cumulants.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Mean squared error against empirical E[ReLU(Z)]",
        "",
        "| layer | method | MSE | MAE | win vs Gaussian | failure fraction |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {method} | {mse:.6g} | {mae:.6g} | {win:.3g} | {fail:.3g} |".format(
                layer=row["layer"],
                method=row["method"],
                mse=row["mean_squared_error"],
                mae=row["mean_abs_error"],
                win=row["win_fraction_vs_gaussian"],
                fail=row["failure_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "If saddlepoint does not beat Gaussian/Edgeworth with oracle cumulants, do not wire it into propagation.  If Edgeworth K3/K4 is also unstable, preserve the current Gaussian/K2 update for robustness.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="development")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--saddlepoint-nodes", type=int, default=48)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.neurons <= 0:
        raise SystemExit("--neurons must be positive")
    if args.saddlepoint_nodes <= 0:
        raise SystemExit("--saddlepoint-nodes must be positive")
    args.layers = tuple(sorted(set(args.layers)))
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in __import__("sys").argv]
    rows = select_rows(args.partition, args.limit)
    records: list[dict[str, Any]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "activation_update_oracle_runs.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            try:
                row_records = evaluate_row(row, args)
            except Exception as exc:
                row_records = [
                    {
                        "schema_version": "1.0",
                        "run_id": args.run_id,
                        "created_at_utc": args.created_at,
                        "partition": args.partition,
                        "mlp_id": int(row["mlp_id"]),
                        "mlp_name": str(row["mlp_name"]),
                        "mlp_seed": int(row["mlp_seed"]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ]
            records.extend(row_records)
            for record in row_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"{index}/{len(rows)} mlp_id={row['mlp_id']} completed", flush=True)
    summary_rows = summarize(records, args)
    suffix = args.partition if args.limit is None else f"{args.partition}_limit{args.limit}"
    csv_path = RESULTS / f"activation_update_oracle_{suffix}_summary.csv"
    report_path = REPORTS / f"activation_update_oracle_{suffix}.md"
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
