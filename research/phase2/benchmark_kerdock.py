"""Paired Kerdock/Puffi replication scout; all failed runs remain in results."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP
from estimator import Estimator as Blend

ROOT = Path(__file__).parent
sys.path.insert(0,str(ROOT/"kerdock_baseline"))
spec = importlib.util.spec_from_file_location("kerdock_impl",ROOT/"kerdock_baseline"/"estimator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)
venture_spec = importlib.util.spec_from_file_location("venture_impl",ROOT/"venture_baseline"/"estimator.py")
venture = importlib.util.module_from_spec(venture_spec)
venture_spec.loader.exec_module(venture)
BUDGET = 2**41
PRESETS = {"blend":None,"k4_raw":(4,0),"k8_raw":(8,0),"k16_raw":(16,0),
           "k8_cv1":(8,1),"k8_cv3":(8,3),"k8_cv5":(8,5),"k8_cv9":(8,9),"k4_cv5":(4,5)}
PRESETS.update(covariance=None,venture=None,venture_uncalibrated=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrow",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--limit",default=8,type=int)
    p.add_argument("--methods",nargs="+",choices=PRESETS,default=list(PRESETS))
    args = p.parse_args()
    data = Dataset.from_file(str(args.arrow)).with_format("numpy")
    report = {"dataset":str(args.arrow),"budget":BUDGET,"score_floor":0.1,
              "scope":"Previously exposed public Mini networks; not fresh holdout or private rank estimate",
              "methods":{name:PRESETS[name] for name in args.methods},"records":[],
              "sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in (ROOT/"kerdock_baseline").iterdir() if f.is_file()},
              "estimator_seed":20260926,"threads":4,
              "versions":{name:importlib.metadata.version(name) for name in
                          ("flopscope","whestbench","numpy","datasets")}}
    report["venture_sha256"] = hashlib.sha256((ROOT/"venture_baseline"/"estimator.py").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(4):
        for i in range(min(args.limit,len(data))):
            row = data[i]
            weights = fnp.asarray(row["weights"],dtype=fnp.float32)
            truth = np.asarray(row["all_layer_means"],dtype=np.float64)
            mlp = MLP(weights.shape[1],weights.shape[0],list(weights),seed=20260926)
            for name in args.methods:
                estimator = Blend() if name in ("blend","covariance") else module.Estimator()
                if name=="covariance":
                    estimator.mode = "covariance"
                elif name.startswith("venture"):
                    estimator = venture.Estimator()
                    estimator.setup(SimpleNamespace(submission_dir=ROOT/"venture_baseline",seed=20260926))
                    if name=="venture_uncalibrated":
                        estimator._calibration = None
                elif name!="blend":
                    estimator.setup(SimpleNamespace(submission_dir=ROOT/"kerdock_baseline"))
                    estimator.frames,estimator.anchor_layer = PRESETS[name]
                record = {"mlp_id":int(row["mlp_id"]),"method":name}
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",UserWarning)
                        with flops.BudgetContext(flop_budget=BUDGET,quiet=True) as ctx:
                            prediction = estimator.predict(mlp,BUDGET)
                    wall = time.perf_counter()-start
                    prediction = np.asarray(prediction,dtype=np.float64)
                    if prediction.shape!=truth.shape or not np.isfinite(prediction).all():
                        raise ValueError("invalid output")
                    mse = float(np.mean((prediction[-1]-truth[-1])**2))
                    residual = ctx.summary_dict()["residual_wall_time_s"]
                    valid = ctx.flops_used<=BUDGET and wall<=120 and residual<=.4
                    adjusted = mse*max(.1,ctx.flops_used/BUDGET)
                    record.update(raw_mse=mse,adjusted=adjusted,utilization=ctx.flops_used/BUDGET,
                                  flops=ctx.flops_used,wall_s=wall,residual_s=residual,valid=valid,
                                  cap_aware_score=adjusted if valid else float(np.mean(truth[-1]**2)))
                except Exception as exc:
                    record.update(error=repr(exc),valid=False,cap_aware_score=float(np.mean(truth[-1]**2)))
                report["records"].append(record)
                args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps(record),flush=True)
    report["summary"] = {}
    for name in args.methods:
        records = [r for r in report["records"] if r["method"]==name]
        scored = [r for r in records if "raw_mse" in r]
        report["summary"][name] = {"count":len(records),"failures":sum(not r["valid"] for r in records),
            "cap_aware_score":float(np.mean([r["cap_aware_score"] for r in records])),
            **{key:float(np.mean([r[key] for r in scored])) if scored else None
               for key in ("raw_mse","adjusted","utilization","wall_s","residual_s")}}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2),flush=True)


if __name__=="__main__":
    main()
