"""Fixed exploratory joint-cumulant scout with development-only calibration."""
import argparse
import json
import hashlib
from pathlib import Path
import time
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from joint_cumulant import Estimator

ROOT = Path(__file__).parent
PRESETS = {"k3_h1":(False,1,16,8),"k34_h1":(True,1,16,8),"k34_h2":(True,2,16,8),
           "k34_h1_q4":(True,1,16,4),"k34_h1_q2":(True,1,16,2),"k34_h1_q1":(True,1,16,1),"k34_prefix4_q4":(True,1,4,4),"k34_prefix8_q4":(True,1,8,4),
           "k34_prefix12_q4":(True,1,12,4)}
BUDGET = 2**41


def fit(records):
    coefficients = []
    for layer in range(16):
        x = np.concatenate([r["design"][layer] for r in records])
        y = np.concatenate([r["truth"][layer]-r["prediction"][layer] for r in records])
        scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-10)
        a = x/scale
        coefficients.append(np.linalg.solve(a.T@a/len(a)+1e-3*np.eye(x.shape[1]),a.T@y/len(a))/scale)
    return np.array(coefficients)


def metrics(net,coef):
    pred = net["prediction"]
    fitted = pred+np.einsum('lnk,lk->ln',net["design"],coef)
    truth = net["truth"]
    return {"mlp_id":net["mlp_id"],"raw_mse_by_layer":np.mean((pred-truth)**2,axis=-1).tolist(),
            "fitted_mse_by_layer":np.mean((fitted-truth)**2,axis=-1).tolist(),
            "nominal_raw_adjusted":float(np.mean((pred[-1]-truth[-1])**2))*max(.1,net["flops"]/BUDGET),
            "nominal_fitted_adjusted":float(np.mean((fitted[-1]-truth[-1])**2))*max(.1,net["flops"]/BUDGET),
            **{k:net[k] for k in ("flops","residual_s","wall_s","resource_valid")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--development-limit",type=int,default=2)
    p.add_argument("--validation-limit",type=int,default=4)
    p.add_argument("--methods",nargs="+",choices=PRESETS,default=list(PRESETS))
    args = p.parse_args()
    report = {"date":"2026-10-01","presets":{k:PRESETS[k] for k in args.methods},
              "protocol":"Two exposed development networks, four disjoint exposed evaluation networks; fixed ridge 0.001, five output-only features per layer. No oracle moments in inference.",
              "source_sha256":hashlib.sha256((ROOT/"joint_cumulant.py").read_bytes()).hexdigest(),
              "records":{"development":{},"validation":{}},"coefficients_frozen":{},
              "caveat":"Fitted readouts evaluated offline; timing is for instrumented raw predictor. Final promotion requires metered fitted prediction and official subprocess/package validation. Invalid timings are never valid scores."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save = lambda:args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save()
    coefs = {}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        for split,path,limit in (("development",args.development,args.development_limit),
                                  ("validation",args.validation,args.validation_limit)):
            dataset = Dataset.from_file(str(path)).with_format("numpy")
            nets = {method:[] for method in args.methods}
            for i in range(min(limit,len(dataset))):
                row = dataset[i]
                w = fnp.asarray(row["weights"],dtype=fnp.float32)
                mlp = MLP(w.shape[1],len(w),list(w),seed=20261001)
                truth = np.asarray(row["all_layer_means"],dtype=np.float64)
                for name in args.methods:
                    est = Estimator()
                    est.use_fourth,est.history,est.rich_layers,est.quadrature_order = PRESETS[name]
                    est.collect_diagnostics = True
                    start = time.perf_counter()
                    try:
                        with flops.BudgetContext(flop_budget=BUDGET,quiet=True) as budget:
                            prediction = est.predict(mlp,BUDGET)
                        wall = time.perf_counter()-start
                        residual = budget.summary_dict()["residual_wall_time_s"]
                        prediction = np.asarray(prediction,dtype=np.float64)
                        assert prediction.shape==truth.shape and np.isfinite(prediction).all()
                        trace = np.array([[np.asarray(x,dtype=np.float64) for x in layer] for layer in est.trace])
                        # trace: layer x (premean,prevariance,K3,K4,postmean,covmass) x neuron.
                        design = np.stack((prediction,trace[:,1],trace[:,0]/np.sqrt(trace[:,1]),
                                           trace[:,5],np.ones_like(prediction)),axis=-1)
                        net = {"mlp_id":int(row["mlp_id"]),"prediction":prediction,"truth":truth,"design":design,
                               "flops":budget.flops_used,"residual_s":residual,"wall_s":wall,
                               "resource_valid":bool(residual<=.4 and wall<=120 and budget.flops_used<=BUDGET)}
                        nets[name].append(net)
                        np.savez_compressed(args.output.with_name(args.output.stem+f"_{name}_mlp{net['mlp_id']}.npz"),
                                            prediction=prediction,truth=truth,design=design,trace=trace)
                        coef = coefs[name] if split=="validation" else np.zeros((16,5))
                        rec = metrics(net,coef)
                        report["records"][split].setdefault(name,[]).append(rec)
                        print(json.dumps({"split":split,"mlp_id":net["mlp_id"],"method":name,
                              "raw_mse":rec["raw_mse_by_layer"][-1],"fitted_mse":rec["fitted_mse_by_layer"][-1],
                              "flops":net["flops"],"residual_s":residual,"wall_s":wall,"valid":net["resource_valid"]}),flush=True)
                    except Exception as exc:
                        rec = {"mlp_id":int(row["mlp_id"]),"error":repr(exc),"resource_valid":False}
                        report["records"][split].setdefault(name,[]).append(rec)
                        print(json.dumps(rec),flush=True)
                    save()
            if split=="development":
                for name in args.methods:
                    if len(nets[name])!=min(limit,len(dataset)):
                        raise RuntimeError(f"Cannot fit {name}: incomplete development predictions")
                    coefs[name] = fit(nets[name])
                    report["coefficients_frozen"][name] = coefs[name].tolist()
                    report["records"][split][name] = [metrics(net,coefs[name]) for net in nets[name]]
                save()
    for split,by_method in report["records"].items():
        report.setdefault("summary",{})[split] = {}
        for name,records in by_method.items():
            ok = [r for r in records if "error" not in r]
            report["summary"][split][name] = {"count":len(records),"failures":sum(not r["resource_valid"] for r in records),
                "raw_mse":float(np.mean([r["raw_mse_by_layer"][-1] for r in ok])),
                "fitted_mse":float(np.mean([r["fitted_mse_by_layer"][-1] for r in ok])),
                **{k:float(np.mean([r[k] for r in ok])) for k in ("nominal_raw_adjusted","nominal_fitted_adjusted","flops","residual_s","wall_s")}}
    save()
    print(json.dumps(report["summary"],indent=2),flush=True)


if __name__=="__main__":
    main()
