"""One-shot full validation of frozen corrected K2."""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, load_estimator
from run_global_corrected_k2 import BASELINE, evaluate, summarize

COEFFICIENT = 0.9918858550287327
OUTPUT = RESULTS / "global_corrected_k2_full_validation.json"
PARTIAL = RESULTS / "global_corrected_k2_full_validation.partial.json"
CHECKPOINT = RESULTS / "global_corrected_k2_full_validation.records.jsonl"
PROGRESS_LOG = RESULTS / "global_corrected_k2_full_validation.progress.log"
FATAL_LOG = RESULTS / "global_corrected_k2_full_validation.fatal.log"


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def result_document(records: list[dict[str, Any]], failures: list[dict[str, Any]], status: str) -> dict[str, Any]:
    result = {"schema_version": "1.0", "experiment": "global_corrected_k2_full_validation"}
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["status"] = status
    result["dataset"], result["revision"], result["split"] = DATASET, REVISION, "full"
    result["coefficient"], result["flop_budget"] = COEFFICIENT, BUDGET
    result["summary"] = summarize(records)
    result["per_mlp"] = records
    result["failures"] = failures
    return result


def load_checkpoint() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if PARTIAL.exists():
        partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
        failures = partial.get("failures", [])
    else:
        failures = []
    return records, failures


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    if not CHECKPOINT.exists():
        CHECKPOINT.write_text("", encoding="utf-8")
        PROGRESS_LOG.write_text("", encoding="utf-8")
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="full"))
    estimator = load_estimator(BASELINE)()
    records, failures = load_checkpoint()
    seen_mlp_ids = {int(record["mlp_id"]) for record in records}
    if records:
        with PROGRESS_LOG.open("a", encoding="utf-8") as stream:
            stream.write(f"resuming after {len(records)}/{len(rows)} records\n")
    for index, row in enumerate(rows, start=1):
        if int(row["mlp_id"]) in seen_mlp_ids:
            continue
        try:
            record = evaluate(estimator, row, COEFFICIENT)
        except Exception as exc:
            record = {
                "mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]), "status": "failed",
            }
            failures.append({
                "mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]),
                "error_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        records.append(record)
        with CHECKPOINT.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(PARTIAL, result_document(records, failures, "running"))
        line = f"{index}/{len(rows)} {record['mlp_name']} {record['status']}"
        with PROGRESS_LOG.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        try:
            print(line, flush=True)
        except OSError:
            pass
    status = "completed" if not failures else "completed_with_failures"
    write_json(OUTPUT, result_document(records, failures, status))
    try:
        print(f"wrote {OUTPUT}", flush=True)
    except OSError:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RESULTS.mkdir(exist_ok=True)
        FATAL_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise
