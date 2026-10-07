"""CLI research experiment, not a submission: oracle shape fed through mean updates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope.numpy as fnp
from whestbench import MLP
from diagnose_distribution import closure,gaussian_relu,risk_summary,source,SOURCE
from diagnose_higher_moments import cumulants,corrections,second_raw_correction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--oracles",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--joint-diagonal-only",action="store_true")
    args = p.parse_args()
    oracle_report = json.loads(args.oracles.read_text(encoding="utf-8"))
    chosen = {r["mlp_id"] for r in oracle_report["records"]["validation"]}
    assert len(chosen)==oracle_report["limit_per_split"], "Complete the oracle experiment first"
    count = 2*oracle_report["samples_per_stream"]
    report = {"protocol":"Two independent pooled streams, fixed analytical K3/K34, no clipping or fitting; mean feedback with unchanged Gaussian covariance update",
              "caveat":"Oracle standardized cumulants, not cheaply propagated cumulants. Gaussian covariance is retained as an ablation, not claimed to be a consistent higher-order closure.",
              "samples_per_stream":count,"records":[]}
    if args.joint_diagonal_only:
        report["protocol"] = "Exploratory follow-up: oracle K3/K4 mean plus diagonal variance correction; Gaussian off-diagonal covariance retained"
        report["caveat"] = "Selected after examining mean-only feedback on this same evaluation set; not a new untouched validation. Oracle moments remain unbudgeted."
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        data = Dataset.from_file(str(args.validation)).with_format("numpy")
        for row in data:
            mid = int(row["mlp_id"])
            if mid not in chosen:
                continue
            start = time.perf_counter()
            w = np.asarray(row["weights"],dtype=np.float32)
            net = np.load(args.oracles.with_name(args.oracles.stem+f"_mlp{mid}.npz"))
            truth = net["truth"]
            baseline = closure(w)["mean"]
            est = source.Estimator()
            est.setup(SimpleNamespace(submission_dir=SOURCE.parent,seed=20261001))
            calibrated = np.asarray(est.predict(MLP(w.shape[1],len(w),[fnp.asarray(x) for x in w],seed=20261001),2**41))
            names = (("k34_mean_variance_feedback",) if args.joint_diagonal_only else
                     ("k3_feedback","k34_feedback","reference_local_bias_feedback"))
            predictions = {name:[] for name in names}
            for j in (0,2):
                v,k3,k4 = cumulants(net["powers"][j:j+2].mean(0),count)
                skew = k3/v**1.5
                excess = k4/v**2
                # The first preactivation is exactly Gaussian: known zero cumulants.
                skew[0] = 0.
                excess[0] = 0.
                delta = truth-gaussian_relu(net["premean"],v)
                delta[0] = 0.
                for method in predictions:
                    def correction(m,var,layer):
                        if method=="reference_local_bias_feedback":
                            return delta[layer]
                        terms = corrections(m,var,skew[layer]*var**1.5,excess[layer]*var**2)
                        return terms[...,0] if method=="k3_feedback" else terms[...,:2].sum(-1)
                    second = (lambda m,var,layer:second_raw_correction(m,var,skew[layer],excess[layer])) if args.joint_diagonal_only else None
                    predictions[method].append(closure(w,mean_correction=correction,second_moment_correction=second)["mean"])
            rec = {"mlp_id":mid,"baseline_mse":np.mean((baseline-truth)**2,axis=-1).tolist(),
                   "calibrated_mse":np.mean((calibrated-truth)**2,axis=-1).tolist(),
                   "methods":{name:risk_summary(pred,truth) for name,pred in predictions.items()}}
            report["records"].append(rec)
            args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps({"mlp_id":mid,"seconds":time.perf_counter()-start,
                "final_cross_risk":{name:r["cross_stream_risk"][-1] for name,r in rec["methods"].items()}}),flush=True)
    report["summary"] = {name:{metric:np.mean([r["methods"][name][metric] for r in report["records"]],axis=0).tolist()
        for metric in ("cross_stream_risk","pooled_prediction_mse","mean_single_stream_mse")}
        for name in report["records"][0]["methods"]}
    for name in ("baseline_mse","calibrated_mse"):
        report["summary"][name] = np.mean([r[name] for r in report["records"]],axis=0).tolist()
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
