"""Summarize the saved offline diagnostic without rerunning any MLPs."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report["records"]
    if not records:
        return {"completed_networks": 0}
    result = {"completed_networks": len(records),
              "maximum_replica_discrepancy": max(r["replica_max_abs_difference"] for r in records)}
    for key in ("baseline_raw_mse_by_layer", "calibrated_raw_mse_by_layer"):
        result[key] = np.mean([r[key] for r in records], axis=0).tolist()
    for group in ("interventions", "restarts"):
        result[group] = {}
        for name in records[0][group]:
            result[group][name] = {
                key: np.mean([r[group][name][key] for r in records], axis=0).tolist()
                for key in ("cross_stream_risk", "pooled_prediction_mse", "mean_single_stream_mse")}
            result[group][name]["per_network_final_cross_risk"] = [
                r[group][name]["cross_stream_risk"][-1] for r in records]
    if all("both_moments_double_samples" in r for r in records):
        result["both_moments_double_samples"] = {
            key: np.mean([r["both_moments_double_samples"][key] for r in records], axis=0).tolist()
            for key in ("cross_stream_risk", "pooled_prediction_mse", "mean_single_stream_mse")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.path)
    formatted = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(formatted + "\n", encoding="utf-8")
    print(formatted)
