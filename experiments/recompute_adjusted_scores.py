"""Update only derived local score fields after a scorer formula correction."""
from __future__ import annotations

import json
from pathlib import Path

from run_baselines import RESULTS, aggregate, table

SCORE_FLOOR = 0.1
BASELINES = {"starter", "mean", "covariance"}


def update(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["configuration"]["baseline"] not in BASELINES:
        return
    result["configuration"]["score_floor"] = SCORE_FLOOR
    done = [record for record in result["per_mlp"] if record["status"] == "completed"]
    for record in done:
        compute, metrics = record["compute"], record["metrics"]
        compute["adjusted_final_score"] = metrics["final_mse"] * max(SCORE_FLOOR, compute["flops_used"] / compute["flop_budget"])
    result["summary"]["adjusted_final_score"] = aggregate(
        record["compute"]["adjusted_final_score"] for record in done
    )
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    paths = []
    for path in sorted(RESULTS.glob("*_mini_*.json")):
        update(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["configuration"]["baseline"] in BASELINES:
            paths.append(path)
    (RESULTS / "baseline_table_mini.md").write_text(table(paths), encoding="utf-8")
    print(f"Corrected {len(paths)} baseline result records to score floor {SCORE_FLOOR}.")


if __name__ == "__main__":
    main()
