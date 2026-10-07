"""Fit and evaluate our Puffi-reference reconstruction on disjoint public shards.

The 90-column linear design follows published equation A28. Unpublished h3 and
published-zero h5 are removed, leaving 72 fitted columns. Ridge is selected by
leave-one-network-out error on the development shard. Validation targets are
loaded only after the final coefficients and ridge value have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP

from estimator import covariance_path
from puffi_reference import Estimator

BUDGET = 2**41
LAYERS = 9
KEEP = np.array([i*10+j for i in range(LAYERS) for j in range(10) if j not in (2,4)])
RIDGES = (1e-6, 1e-4, 1e-2, 1.0)


def design_at_layer(trace, weights, target_layer):
    """Return [width, 90] mapping beta to transported correction at target."""
    blocks = []
    for source in range(LAYERS):
        block = np.asarray(trace[source][1], dtype=np.float64)
        for layer in range(source+1, target_layer+1):
            gain = np.asarray(trace[layer][2], dtype=np.float64)
            block = gain[:,None] * (np.asarray(weights[layer], dtype=np.float64).T @ block)
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def fit(train, alpha):
    x = np.concatenate([r["design9"][:,KEEP] for r in train])
    y = np.concatenate([r["residual9"] for r in train])
    scale = np.sqrt(np.mean(x*x, axis=0))
    scale = np.maximum(scale, 1e-14)
    xs = x/scale
    gram = xs.T @ xs
    ridge = alpha * float(np.trace(gram)/gram.shape[0])
    coef = np.linalg.solve(gram + ridge*np.eye(gram.shape[0]), xs.T @ y)
    return coef/scale


def expand_beta(compact):
    full = np.zeros(LAYERS*10, dtype=np.float64)
    full[KEEP] = compact
    return full.reshape(LAYERS,10)


def load_records(path, label):
    dataset = Dataset.from_file(str(path)).with_format("numpy")
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        weights = fnp.asarray(row["weights"], dtype=fnp.float32)
        mlp = MLP(1024,16,[weights[i] for i in range(16)],seed=20260905)
        estimator = Estimator()
        estimator.collect_diagnostics = True
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with flops.BudgetContext(flop_budget=BUDGET,quiet=True) as context:
                prediction = estimator.predict(mlp,BUDGET)
        wall = time.perf_counter()-started
        base = np.asarray(prediction,dtype=np.float64)
        target = np.asarray(row["all_layer_means"],dtype=np.float64)
        d9 = design_at_layer(estimator.trace,row["weights"],8)
        d16 = design_at_layer(estimator.trace,row["weights"],15)
        # Separate covariance run is a comparison, never used to fit beta.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with flops.BudgetContext(flop_budget=BUDGET,quiet=True) as cov_context:
                covariance = covariance_path(mlp)[0]
        covariance = np.asarray(covariance,dtype=np.float64)
        rec = {
            "label": label, "mlp_id": int(row["mlp_id"]), "base": base,
            "target": target, "design9": d9, "design16": d16,
            "residual9": base[8]-target[8], "flops": context.flops_used,
            "utilization": context.flops_used/BUDGET, "wall_s": wall,
            "residual_s": context.summary_dict()["residual_wall_time_s"],
            "covariance9_mse": float(np.mean((covariance[8]-target[8])**2)),
            "covariance16_mse": float(np.mean((covariance[15]-target[15])**2)),
            "covariance_flops": cov_context.flops_used,
        }
        records.append(rec)
        print(label,index,{k:rec[k] for k in ("mlp_id","utilization","wall_s","residual_s")},flush=True)
    return records


def score(records,beta):
    compact=beta.reshape(-1)[KEEP]
    result=[]
    for r in records:
        p9=np.maximum(r["base"][8]-r["design9"][:,KEEP]@compact,0)
        p16=np.maximum(r["base"][15]-r["design16"][:,KEEP]@compact,0)
        result.append({
            "mlp_id":r["mlp_id"],
            "reference9_mse":float(np.mean((r["base"][8]-r["target"][8])**2)),
            "fitted9_mse":float(np.mean((p9-r["target"][8])**2)),
            "reference16_mse":float(np.mean((r["base"][15]-r["target"][15])**2)),
            "fitted16_mse":float(np.mean((p16-r["target"][15])**2)),
            "covariance9_mse":r["covariance9_mse"],
            "covariance16_mse":r["covariance16_mse"],
            "flops":r["flops"],"utilization":r["utilization"],
            "wall_s":r["wall_s"],"residual_s":r["residual_s"],
        })
    fields=("reference9_mse","fitted9_mse","reference16_mse","fitted16_mse",
            "covariance9_mse","covariance16_mse","utilization","wall_s","residual_s")
    summary={f"mean_{f}":float(np.mean([x[f] for x in result])) for f in fields}
    summary["adjusted_fitted16"]=summary["mean_fitted16_mse"]*max(0.1,summary["mean_utilization"])
    summary["adjusted_reference16"]=summary["mean_reference16_mse"]*max(0.1,summary["mean_utilization"])
    summary["adjusted_covariance16"]=summary["mean_covariance16_mse"]*0.1
    return result,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    with threadpool_limits(args.threads):
        development=load_records(args.development,"development")
        cv={}
        for alpha in RIDGES:
            values=[]
            for held in range(len(development)):
                beta=expand_beta(fit([r for i,r in enumerate(development) if i!=held],alpha))
                _,s=score([development[held]],beta)
                values.append(s["mean_fitted9_mse"])
            cv[str(alpha)]={"mean_layer9_mse":float(np.mean(values)),"per_network":values}
        selected=min(RIDGES,key=lambda x:cv[str(x)]["mean_layer9_mse"])
        beta=expand_beta(fit(development,selected))
        # Freeze before reading the validation shard.
        beta_sha=hashlib.sha256(beta.astype("<f8").tobytes()).hexdigest()
        development_scores,development_summary=score(development,beta)
        validation=load_records(args.validation,"validation")
        validation_scores,validation_summary=score(validation,beta)
    report={
        "protocol":json.loads(Path(__file__).with_name("puffi_protocol.json").read_text()),
        "implementation_sha256":hashlib.sha256(Path(__file__).with_name("puffi_reference.py").read_bytes()).hexdigest(),
        "ridge_cv":cv,"selected_ridge":selected,"beta":beta.tolist(),"beta_sha256":beta_sha,
        "development":{"summary":development_summary,"per_mlp":development_scores},
        "validation":{"summary":validation_summary,"per_mlp":validation_scores},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"selected_ridge":selected,"beta_sha256":beta_sha,
                      "development":development_summary,"validation":validation_summary},indent=2))


if __name__=="__main__":
    main()
