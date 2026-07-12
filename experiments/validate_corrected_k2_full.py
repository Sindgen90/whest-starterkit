"""One-shot full validation of frozen corrected K2."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, load_estimator
from run_global_corrected_k2 import BASELINE, evaluate, summarize

COEFFICIENT = 0.9918858550287327


def main() -> None:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="full"))
    estimator = load_estimator(BASELINE)()
    records, failures = [], []
    for row in rows:
        records.append(evaluate(estimator, row, COEFFICIENT))
    result = {"experiment": "global_corrected_k2_full_validation"}
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["dataset"], result["revision"], result["split"] = DATASET, REVISION, "full"
    result["coefficient"], result["flop_budget"] = COEFFICIENT, BUDGET
    result["summary"] = summarize(records)
    result["per_mlp"] = records
    result["failures"] = failures
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "global_corrected_k2_full_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
