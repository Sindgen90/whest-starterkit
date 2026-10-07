"""Exploratory oracle-variance control after K3/K4 feedback diagnostics."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
from diagnose_distribution import gaussian_relu,risk_summary
from diagnose_higher_moments import cumulants,corrections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--oracles",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    original = json.loads(args.oracles.read_text(encoding="utf-8"))
    ids = {r["mlp_id"] for r in original["records"]["validation"]}
    count = 2*original["samples_per_stream"]
    report = {"protocol":"Adaptive diagnostic on already exposed evaluation networks: recursively propagate means while supplying sampled true incoming variances at every layer. Compare Gaussian and oracle K3+K4 updates.",
              "caveat":"No propagated covariance. This is not an estimator or unbiased population oracle. Two independent pooled streams; first-layer moments known exactly.","records":[]}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        for row in Dataset.from_file(str(args.validation)).with_format("numpy"):
            mid = int(row["mlp_id"])
            if mid not in ids:
                continue
            w = np.asarray(row["weights"],dtype=np.float32)
            net = np.load(args.oracles.with_name(args.oracles.stem+f"_mlp{mid}.npz"))
            predictions = {name:[] for name in ("gaussian_oracle_variance","k34_oracle_variance")}
            for j in (0,2):
                v,k3,k4 = cumulants(net["powers"][j:j+2].mean(0),count)
                v[0] = np.sum(w[0].astype(np.float64)**2,axis=0)
                k3[0] = 0.
                k4[0] = 0.
                for name in predictions:
                    mu = np.zeros(w.shape[1])
                    rows = []
                    for layer,weight in enumerate(w):
                        m = mu@weight
                        mu = gaussian_relu(m,v[layer])
                        if name=="k34_oracle_variance":
                            mu += corrections(m,v[layer],k3[layer],k4[layer])[...,:2].sum(-1)
                        rows.append(mu)
                    predictions[name].append(np.array(rows))
            rec = {"mlp_id":mid,"methods":{name:risk_summary(pred,net["truth"]) for name,pred in predictions.items()}}
            report["records"].append(rec)
    report["summary"] = {name:{metric:np.mean([r["methods"][name][metric] for r in report["records"]],axis=0).tolist()
        for metric in ("cross_stream_risk","pooled_prediction_mse","mean_single_stream_mse")}
        for name in report["records"][0]["methods"]}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({name:{metric:values[-1] for metric,values in result.items()} for name,result in report["summary"].items()},indent=2))


if __name__=="__main__":
    main()
