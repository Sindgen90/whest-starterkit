"""Grid a scalar corrected-K2/RQMC convex blend on frozen dev/validation splits.

This is deliberately narrow: no new estimator families, no holdout access, and no
per-MLP fitting.  For each RQMC configuration, the script:

1. caches corrected-K2 predictions;
2. fits one global scalar lambda on development calibration MLPs;
3. evaluates on development selection MLPs;
4. evaluates only the top development configs on validation, with lambda frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
CORRECTED_K2 = ROOT / "submissions" / "corrected_k2" / "estimator.py"
RQMC = ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py"

BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)
DEFAULT_SAMPLE_COUNTS = (4096, 5120)
DEFAULT_PAIRS = (
    (8_675_309, 17_797),
    (8_675_309, 32_771),
    (8_675_309, 271_829),
    (8_675_309, 22_695_477),
    (8_675_309, 747_796_405),
    (8_675_309, 2_891_336_453),
    (314_159, 17_797),
    (271_828, 17_797),
    (1_000_003, 17_797),
    (2_000_003, 17_797),
)


@dataclass(frozen=True)
class Candidate:
    n_samples: int
    seed_offset: int
    multiplier: int

    @property
    def key(self) -> str:
        return f"n{self.n_samples}_offset{self.seed_offset}_mult{self.multiplier}"


@dataclass
class PredictionRecord:
    prediction: np.ndarray
    flops_used: int
    runtime_s: float
    status: str = "completed"
    error_type: str | None = None
    error_message: str | None = None


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_estimator_class(path: Path) -> type[Any]:
    module_name = f"estimator_{path.parent.name}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator


def load_rqmc_estimator(candidate: Candidate) -> Any:
    module_name = f"rqmc_{candidate.key}"
    spec = importlib.util.spec_from_file_location(module_name, RQMC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import estimator from {RQMC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_SAMPLES = candidate.n_samples
    module.RQMC_SEED_OFFSET = candidate.seed_offset
    module.RQMC_MULTIPLIER = candidate.multiplier
    return module.Estimator()


def as_mlp(row: dict[str, Any]) -> SimpleNamespace:
    stacked = fnp.array(np.asarray(row["weights"], dtype=np.float32))
    depth, width, _ = stacked.shape
    return SimpleNamespace(
        width=int(width),
        depth=int(depth),
        weights=[stacked[i] for i in range(depth)],
        seed=int(row["mlp_seed"]),
        name=str(row["mlp_name"]),
    )


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    holdout = set(int(x) for x in manifest["holdout_mlp_ids"])
    if not holdout:
        raise RuntimeError("Holdout split is empty; refusing to run.")
    return manifest


def select_rows(
    rows: list[dict[str, Any]],
    ids: Iterable[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    wanted = set(int(x) for x in ids)
    selected = [row for row in rows if int(row["mlp_id"]) in wanted]
    selected.sort(key=lambda row: int(row["mlp_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def predict(estimator: Any, mlp: SimpleNamespace, budget: int) -> PredictionRecord:
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        prediction = estimator.predict(mlp, budget)
    runtime_s = time.perf_counter() - started
    prediction_np = np.asarray(prediction, dtype=np.float64)
    expected_shape = (mlp.depth, mlp.width)
    if prediction_np.shape != expected_shape:
        raise ValueError(f"prediction shape {prediction_np.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(prediction_np)):
        raise ValueError("prediction contains non-finite values")
    return PredictionRecord(prediction=prediction_np, flops_used=int(context.flops_used), runtime_s=runtime_s)


def blend_predictions(k2: np.ndarray, rqmc: np.ndarray, lambda_: float, budget: int) -> tuple[np.ndarray, int]:
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        combo = fnp.add(
            fnp.multiply(fnp.array(k2), 1.0 - lambda_),
            fnp.multiply(fnp.array(rqmc), lambda_),
        )
    return np.asarray(combo, dtype=np.float64), int(context.flops_used)


def fit_lambda(
    rows: list[dict[str, Any]],
    k2_predictions: dict[int, PredictionRecord],
    rqmc_predictions: dict[int, PredictionRecord],
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    per_mlp = []
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        k2 = k2_predictions[mlp_id].prediction
        rqmc = rqmc_predictions[mlp_id].prediction
        delta = rqmc[-1] - k2[-1]
        residual = k2[-1] - target[-1]
        numerator += float(np.vdot(delta, residual))
        denominator += float(np.vdot(delta, delta))
        per_mlp.append(
            {
                "mlp_id": mlp_id,
                "k2_final_mse": final_mse(k2, target),
                "rqmc_final_mse": final_mse(rqmc, target),
            }
        )
    raw_lambda = -numerator / denominator if denominator else 0.0
    lambda_ = min(1.0, max(0.0, raw_lambda))
    return {
        "objective": "development calibration final-layer least squares",
        "formula": "argmin_lambda ||k2 + lambda*(rqmc-k2) - target||^2, clipped to [0,1]",
        "raw_lambda": raw_lambda,
        "lambda": lambda_,
        "n_mlps": len(rows),
        "records": per_mlp,
    }


def final_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(((prediction[-1] - target[-1]) ** 2).mean())


def all_layer_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(((prediction - target) ** 2).mean())


def prefix_mse(prediction: np.ndarray, target: np.ndarray, prefixes: Iterable[int]) -> dict[str, float]:
    return {
        str(prefix): float(((prediction[prefix - 1] - target[prefix - 1]) ** 2).mean())
        for prefix in prefixes
        if prefix <= prediction.shape[0]
    }


def aggregate(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return {"mean": None, "median": None, "standard_error": None, "worst_decile_mean": None, "max": None}
    worst_count = max(1, math.ceil(len(data) / 10))
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "standard_error": float(data.std(ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0,
        "worst_decile_mean": float(np.sort(data)[-worst_count:].mean()),
        "max": float(data.max()),
    }


def evaluate_split(
    split_name: str,
    rows: list[dict[str, Any]],
    candidate: Candidate,
    lambda_: float,
    k2_predictions: dict[int, PredictionRecord],
    rqmc_predictions: dict[int, PredictionRecord],
    budget: int,
    prefixes: Iterable[int],
) -> dict[str, Any]:
    records = []
    failures = []
    for row in rows:
        mlp_id = int(row["mlp_id"])
        try:
            target = np.asarray(row["all_layer_means"], dtype=np.float64)
            k2 = k2_predictions[mlp_id]
            rqmc = rqmc_predictions[mlp_id]
            combo_prediction, blend_flops = blend_predictions(k2.prediction, rqmc.prediction, lambda_, budget)
            combo_flops = k2.flops_used + rqmc.flops_used + blend_flops
            combo_final_mse = final_mse(combo_prediction, target)
            k2_final_mse = final_mse(k2.prediction, target)
            rqmc_final_mse = final_mse(rqmc.prediction, target)
            records.append(
                {
                    "mlp_id": mlp_id,
                    "mlp_name": str(row["mlp_name"]),
                    "mlp_seed": int(row["mlp_seed"]),
                    "status": "completed",
                    "metrics": {
                        "corrected_k2": {
                            "final_mse": k2_final_mse,
                            "all_layer_mse": all_layer_mse(k2.prediction, target),
                            "prefix_mse": prefix_mse(k2.prediction, target, prefixes),
                        },
                        "rqmc": {
                            "final_mse": rqmc_final_mse,
                            "all_layer_mse": all_layer_mse(rqmc.prediction, target),
                            "prefix_mse": prefix_mse(rqmc.prediction, target, prefixes),
                        },
                        "combo": {
                            "final_mse": combo_final_mse,
                            "all_layer_mse": all_layer_mse(combo_prediction, target),
                            "prefix_mse": prefix_mse(combo_prediction, target, prefixes),
                        },
                    },
                    "paired_vs_rqmc": {
                        "combo_final_mse_delta": combo_final_mse - rqmc_final_mse,
                        "combo_final_mse_ratio": combo_final_mse / rqmc_final_mse if rqmc_final_mse else None,
                        "combo_win": combo_final_mse < rqmc_final_mse,
                    },
                    "paired_vs_corrected_k2": {
                        "combo_final_mse_delta": combo_final_mse - k2_final_mse,
                        "combo_final_mse_ratio": combo_final_mse / k2_final_mse if k2_final_mse else None,
                        "combo_win": combo_final_mse < k2_final_mse,
                    },
                    "compute": {
                        "corrected_k2_flops": k2.flops_used,
                        "rqmc_flops": rqmc.flops_used,
                        "blend_flops": blend_flops,
                        "combo_flops": combo_flops,
                        "flop_budget": budget,
                        "combo_flop_fraction": combo_flops / budget,
                        "corrected_k2_runtime_s": k2.runtime_s,
                        "rqmc_runtime_s": rqmc.runtime_s,
                        "combo_runtime_s": k2.runtime_s + rqmc.runtime_s,
                        "combo_adjusted_final_score": combo_final_mse * max(SCORE_FLOOR, combo_flops / budget),
                    },
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "mlp_id": mlp_id,
                    "mlp_name": str(row.get("mlp_name", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            records.append(
                {
                    "mlp_id": mlp_id,
                    "mlp_name": str(row.get("mlp_name", "")),
                    "mlp_seed": int(row.get("mlp_seed", -1)),
                    "status": "failed",
                }
            )
    return {
        "split": split_name,
        "candidate": {
            "key": candidate.key,
            "n_samples": candidate.n_samples,
            "seed_offset": candidate.seed_offset,
            "multiplier": candidate.multiplier,
        },
        "lambda": lambda_,
        "summary": summarize_records(records, prefixes),
        "per_mlp": records,
        "failures": failures,
    }


def summarize_records(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    summary: dict[str, Any] = {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "corrected_k2_final_mse": aggregate(r["metrics"]["corrected_k2"]["final_mse"] for r in done),
        "rqmc_final_mse": aggregate(r["metrics"]["rqmc"]["final_mse"] for r in done),
        "combo_final_mse": aggregate(r["metrics"]["combo"]["final_mse"] for r in done),
        "corrected_k2_all_layer_mse": aggregate(r["metrics"]["corrected_k2"]["all_layer_mse"] for r in done),
        "rqmc_all_layer_mse": aggregate(r["metrics"]["rqmc"]["all_layer_mse"] for r in done),
        "combo_all_layer_mse": aggregate(r["metrics"]["combo"]["all_layer_mse"] for r in done),
        "combo_adjusted_final_score": aggregate(r["compute"]["combo_adjusted_final_score"] for r in done),
        "combo_flops": aggregate(r["compute"]["combo_flops"] for r in done),
        "combo_runtime_s": aggregate(r["compute"]["combo_runtime_s"] for r in done),
        "paired_vs_rqmc": {
            "combo_win_fraction": float(np.mean([r["paired_vs_rqmc"]["combo_win"] for r in done])) if done else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_rqmc"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(
                r["paired_vs_rqmc"]["combo_final_mse_ratio"]
                for r in done
                if r["paired_vs_rqmc"]["combo_final_mse_ratio"] is not None
            ),
        },
        "paired_vs_corrected_k2": {
            "combo_win_fraction": float(np.mean([r["paired_vs_corrected_k2"]["combo_win"] for r in done])) if done else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(
                r["paired_vs_corrected_k2"]["combo_final_mse_ratio"]
                for r in done
                if r["paired_vs_corrected_k2"]["combo_final_mse_ratio"] is not None
            ),
        },
        "prefix_mse": {},
    }
    for prefix in prefixes:
        key = str(prefix)
        summary["prefix_mse"][key] = {
            "corrected_k2": aggregate(
                r["metrics"]["corrected_k2"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["corrected_k2"]["prefix_mse"]
            ),
            "rqmc": aggregate(
                r["metrics"]["rqmc"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["rqmc"]["prefix_mse"]
            ),
            "combo": aggregate(
                r["metrics"]["combo"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["combo"]["prefix_mse"]
            ),
        }
    return summary


def flatten_summary(result: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    candidate = result["candidate"]
    return {
        "split": result["split"],
        "candidate_key": candidate["key"],
        "n_samples": candidate["n_samples"],
        "seed_offset": candidate["seed_offset"],
        "multiplier": candidate["multiplier"],
        "lambda": result["lambda"],
        "raw_lambda": calibration["raw_lambda"],
        "n_completed": summary["n_completed"],
        "n_failed": summary["n_failed"],
        "k2_final_mse_mean": summary["corrected_k2_final_mse"]["mean"],
        "rqmc_final_mse_mean": summary["rqmc_final_mse"]["mean"],
        "combo_final_mse_mean": summary["combo_final_mse"]["mean"],
        "combo_final_mse_median": summary["combo_final_mse"]["median"],
        "combo_final_mse_se": summary["combo_final_mse"]["standard_error"],
        "combo_final_mse_worst_decile": summary["combo_final_mse"]["worst_decile_mean"],
        "combo_final_mse_max": summary["combo_final_mse"]["max"],
        "combo_adjusted_score_mean": summary["combo_adjusted_final_score"]["mean"],
        "combo_flops_mean": summary["combo_flops"]["mean"],
        "combo_win_fraction_vs_rqmc": summary["paired_vs_rqmc"]["combo_win_fraction"],
        "combo_delta_vs_rqmc_mean": summary["paired_vs_rqmc"]["combo_final_mse_delta"]["mean"],
        "combo_ratio_vs_rqmc_mean": summary["paired_vs_rqmc"]["combo_final_mse_ratio"]["mean"],
        "combo_win_fraction_vs_k2": summary["paired_vs_corrected_k2"]["combo_win_fraction"],
        "combo_delta_vs_k2_mean": summary["paired_vs_corrected_k2"]["combo_final_mse_delta"]["mean"],
        "combo_ratio_vs_k2_mean": summary["paired_vs_corrected_k2"]["combo_final_mse_ratio"]["mean"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    args: argparse.Namespace,
    dev_rows: list[dict[str, Any]],
    dev_calibration_rows: list[dict[str, Any]],
    dev_selection_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    dev_summary_rows: list[dict[str, Any]],
    validation_summary_rows: list[dict[str, Any]],
    top_candidates: list[str],
    command: str,
    run_id: str,
) -> None:
    best_dev = sorted(dev_summary_rows, key=lambda row: row["combo_final_mse_mean"])[:5]
    best_val = sorted(validation_summary_rows, key=lambda row: row["combo_final_mse_mean"]) if validation_summary_rows else []
    columns = [
        ("candidate", "candidate_key"),
        ("lambda", "lambda"),
        ("combo MSE", "combo_final_mse_mean"),
        ("RQMC MSE", "rqmc_final_mse_mean"),
        ("K2 MSE", "k2_final_mse_mean"),
        ("adj score", "combo_adjusted_score_mean"),
        ("FLOPs", "combo_flops_mean"),
        ("win vs RQMC", "combo_win_fraction_vs_rqmc"),
    ]
    report = [
        "# K2 + RQMC convex-combo grid",
        "",
        f"Run ID: `{run_id}`",
        "",
        "This experiment stayed inside the narrow fallback lane: corrected K2 plus ordinary RQMC, with one globally calibrated convex weight.  No holdout MLPs were evaluated.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{command}\n```",
        "",
        "## Split policy",
        "",
        f"- Development rows loaded: {len(dev_rows)}",
        f"- Development calibration rows: {len(dev_calibration_rows)}",
        f"- Development selection rows: {len(dev_selection_rows)}",
        f"- Validation rows: {len(validation_rows)}",
        f"- Holdout rows touched: 0",
        "",
        "## Development top 5",
        "",
        markdown_table(best_dev, columns),
        "",
        "## Validation results for frozen top configs",
        "",
        markdown_table(best_val, columns) if best_val else "_Validation was not run._",
        "",
        "## Decision",
        "",
    ]
    if best_val:
        winner = best_val[0]
        report.extend(
            [
                f"Best validation candidate: `{winner['candidate_key']}` with lambda `{format_float(winner['lambda'])}`.",
                f"Validation combo final MSE: `{format_float(winner['combo_final_mse_mean'])}`; adjusted score: `{format_float(winner['combo_adjusted_score_mean'])}`.",
                "",
                "This is a candidate worth packaging only if it beats the already packaged K2+Stein/RQMC submission on the same validation protocol.  If it does not, keep this as a negative-result grid and do not broaden the method.",
            ]
        )
    else:
        report.append("No validation winner; use the development grid only as a smoke/diagnostic run.")
    report.extend(
        [
            "",
            "## Files",
            "",
            f"- Dev summary: `research/results/k2_rqmc_combo_grid_dev_summary.csv`",
            f"- Validation summary: `research/results/k2_rqmc_combo_grid_validation_summary.csv`",
            f"- Full JSONL log: `research/results/k2_rqmc_combo_grid.jsonl`",
            "",
            f"Top candidates sent to validation: {', '.join(f'`{key}`' for key in top_candidates)}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_pair(text: str) -> tuple[int, int]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("pairs must be seed_offset:multiplier")
    left, right = text.split(":", 1)
    seed_offset = int(left.replace("_", ""))
    multiplier = int(right.replace("_", ""))
    if seed_offset < 0:
        raise argparse.ArgumentTypeError("seed_offset must be non-negative")
    if multiplier <= 0 or multiplier % 2 == 0:
        raise argparse.ArgumentTypeError("multiplier must be a positive odd integer")
    return seed_offset, multiplier


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = [
        Candidate(n_samples=n, seed_offset=seed_offset, multiplier=multiplier)
        for n in args.sample_counts
        for seed_offset, multiplier in args.pairs
    ]
    seen: set[str] = set()
    deduped = []
    for candidate in candidates:
        if candidate.key not in seen:
            seen.add(candidate.key)
            deduped.append(candidate)
    if args.max_candidates is not None:
        deduped = deduped[: args.max_candidates]
    return deduped


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.prefixes = tuple(sorted(set(args.prefixes)))
    manifest = load_manifest()
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    dev_rows = select_rows(rows, manifest["development_mlp_ids"], args.dev_limit)
    validation_rows = select_rows(rows, manifest["validation_mlp_ids"], args.validation_limit)
    holdout_ids = set(int(x) for x in manifest["holdout_mlp_ids"])
    touched_ids = {int(row["mlp_id"]) for row in [*dev_rows, *validation_rows]}
    if touched_ids & holdout_ids:
        raise RuntimeError(f"Refusing to evaluate holdout IDs: {sorted(touched_ids & holdout_ids)}")

    dev_calibration_rows = [row for row in dev_rows if int(row["mlp_id"]) % 2 == args.calibration_parity]
    dev_selection_rows = [row for row in dev_rows if int(row["mlp_id"]) % 2 != args.calibration_parity]
    if not dev_calibration_rows or not dev_selection_rows:
        raise RuntimeError("Need non-empty development calibration and selection splits.")

    mlp_cache = {int(row["mlp_id"]): as_mlp(row) for row in [*dev_rows, *validation_rows]}
    k2_estimator = load_estimator_class(CORRECTED_K2)()
    k2_predictions: dict[int, PredictionRecord] = {}
    k2_rows = [*dev_rows, *validation_rows]
    print(f"Caching corrected K2 for {len(k2_rows)} MLPs...", flush=True)
    for index, row in enumerate(k2_rows, start=1):
        mlp_id = int(row["mlp_id"])
        try:
            k2_predictions[mlp_id] = predict(k2_estimator, mlp_cache[mlp_id], args.budget)
            print(f"K2 {index}/{len(k2_rows)} mlp_id={mlp_id}", flush=True)
        except Exception as exc:
            k2_predictions[mlp_id] = PredictionRecord(
                prediction=np.empty((0, 0)),
                flops_used=0,
                runtime_s=0.0,
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            print(f"K2 {index}/{len(k2_rows)} mlp_id={mlp_id} failed: {exc}", flush=True)

    candidates = build_candidates(args)
    dev_summary_rows: list[dict[str, Any]] = []
    validation_summary_rows: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []

    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "k2_rqmc_combo_grid.jsonl"
    if not args.append_jsonl:
        jsonl_path.write_text("", encoding="utf-8")
    print(f"Running {len(candidates)} RQMC candidate(s) on development...", flush=True)
    for cand_index, candidate in enumerate(candidates, start=1):
        rqmc_estimator = load_rqmc_estimator(candidate)
        rqmc_predictions: dict[int, PredictionRecord] = {}
        for row in dev_rows:
            mlp_id = int(row["mlp_id"])
            rqmc_predictions[mlp_id] = predict(rqmc_estimator, mlp_cache[mlp_id], args.budget)
        calibration = fit_lambda(dev_calibration_rows, k2_predictions, rqmc_predictions)
        dev_result = evaluate_split(
            "development_selection",
            dev_selection_rows,
            candidate,
            calibration["lambda"],
            k2_predictions,
            rqmc_predictions,
            args.budget,
            args.prefixes,
        )
        row = flatten_summary(dev_result, calibration)
        dev_summary_rows.append(row)
        jsonl_record = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "development_grid",
            "candidate": dev_result["candidate"],
            "calibration": calibration,
            "result": dev_result,
        }
        jsonl_records.append(jsonl_record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonl_record, sort_keys=True) + "\n")
        print(
            f"DEV {cand_index}/{len(candidates)} {candidate.key}: "
            f"lambda={calibration['lambda']:.4f} "
            f"combo_mse={row['combo_final_mse_mean']:.6g} "
            f"rqmc_mse={row['rqmc_final_mse_mean']:.6g}",
            flush=True,
        )

    top_dev = sorted(dev_summary_rows, key=lambda row: row["combo_final_mse_mean"])[: args.top_k]
    top_keys = [row["candidate_key"] for row in top_dev]
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    print(f"Running validation for top {len(top_keys)} candidate(s): {', '.join(top_keys)}", flush=True)
    for row in top_dev:
        candidate = candidate_by_key[row["candidate_key"]]
        rqmc_estimator = load_rqmc_estimator(candidate)
        rqmc_predictions = {
            int(validation_row["mlp_id"]): predict(rqmc_estimator, mlp_cache[int(validation_row["mlp_id"])], args.budget)
            for validation_row in validation_rows
        }
        # Recompute dev calibration for this exact candidate, then keep lambda fixed.
        dev_rqmc_predictions = {
            int(dev_row["mlp_id"]): predict(rqmc_estimator, mlp_cache[int(dev_row["mlp_id"])], args.budget)
            for dev_row in dev_rows
        }
        calibration = fit_lambda(dev_calibration_rows, k2_predictions, dev_rqmc_predictions)
        validation_result = evaluate_split(
            "validation_frozen_lambda",
            validation_rows,
            candidate,
            calibration["lambda"],
            k2_predictions,
            rqmc_predictions,
            args.budget,
            args.prefixes,
        )
        flat = flatten_summary(validation_result, calibration)
        validation_summary_rows.append(flat)
        jsonl_record = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "validation_top",
            "candidate": validation_result["candidate"],
            "calibration": calibration,
            "result": validation_result,
        }
        jsonl_records.append(jsonl_record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonl_record, sort_keys=True) + "\n")
        print(
            f"VAL {candidate.key}: lambda={calibration['lambda']:.4f} "
            f"combo_mse={flat['combo_final_mse_mean']:.6g} "
            f"rqmc_mse={flat['rqmc_final_mse_mean']:.6g}",
            flush=True,
        )

    dev_csv = RESULTS / "k2_rqmc_combo_grid_dev_summary.csv"
    validation_csv = RESULTS / "k2_rqmc_combo_grid_validation_summary.csv"
    write_csv(dev_csv, dev_summary_rows)
    write_csv(validation_csv, validation_summary_rows)
    report_path = REPORTS / "k2_rqmc_combo_grid_report.md"
    write_report(
        report_path,
        args,
        dev_rows,
        dev_calibration_rows,
        dev_selection_rows,
        validation_rows,
        dev_summary_rows,
        validation_summary_rows,
        top_keys,
        " ".join(sys.argv),
        args.run_id,
    )
    return {
        "run_id": args.run_id,
        "dev_csv": str(dev_csv),
        "validation_csv": str(validation_csv),
        "jsonl": str(jsonl_path),
        "report": str(report_path),
        "top_keys": top_keys,
        "best_dev": sorted(dev_summary_rows, key=lambda row: row["combo_final_mse_mean"])[0] if dev_summary_rows else None,
        "best_validation": sorted(validation_summary_rows, key=lambda row: row["combo_final_mse_mean"])[0]
        if validation_summary_rows
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--sample-counts", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument(
        "--pairs",
        type=parse_pair,
        nargs="+",
        default=list(DEFAULT_PAIRS),
        help="RQMC randomization/generator pairs as seed_offset:multiplier.",
    )
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--append-jsonl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if any(n <= 0 for n in args.sample_counts):
        raise SystemExit("--sample-counts must be positive")
    if args.dev_limit is not None and args.dev_limit <= 2:
        raise SystemExit("--dev-limit must be greater than 2")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
