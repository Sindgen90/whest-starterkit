"""Paired accuracy and resource comparison; frozen baseline vs runtime rewrite."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import time
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

ROOT = Path(__file__).parent


def load(folder):
    spec = importlib.util.spec_from_file_location(folder,ROOT/folder/"estimator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrows",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=4)
    args = p.parse_args()
    modules = {name:load(name) for name in ("venture_baseline","venture_runtime")}
    report = {"protocol":"No fitting. Paired four networks per exposed shard; identical eight-node formula and calibration, float32, four threads.","records":[]}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        for path in args.arrows:
            data = Dataset.from_file(str(path)).with_format("numpy")
            for i in range(min(args.limit,len(data))):
                row = data[i]
                w = fnp.asarray(row["weights"],dtype=fnp.float32)
                mlp = MLP(w.shape[1],len(w),list(w),seed=20261001)
                truth = np.asarray(row["all_layer_means"],dtype=np.float64)
                original = None
                for name,mod in modules.items():
                    est = mod.Estimator()
                    est.setup(SimpleNamespace(submission_dir=ROOT/name,seed=20261001))
                    start = time.perf_counter()
                    with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
                        pred = np.asarray(est.predict(mlp,2**41),dtype=np.float64)
                    wall = time.perf_counter()-start
                    assert pred.shape==truth.shape and np.isfinite(pred).all()
                    if original is None:
                        original = pred.copy()
                    diff = float(np.max(np.abs(pred-original)))
                    assert diff<1e-5, f"Unexpected numerical drift: {diff}"
                    residual = budget.summary_dict()["residual_wall_time_s"]
                    mse = float(np.mean((pred[-1]-truth[-1])**2))
                    valid = residual<=.4 and wall<=120 and budget.flops_used<=2**41
                    rec = {"mlp_id":int(row["mlp_id"]),"method":name,"raw_mse":mse,
                           "nominal_adjusted":mse*max(.1,budget.flops_used/2**41),
                           "flops":budget.flops_used,"residual_s":residual,"wall_s":wall,"valid":valid,
                           "cap_aware_score":mse*max(.1,budget.flops_used/2**41) if valid else float(np.mean(truth[-1]**2)),
                           "max_abs_prediction_difference":diff}
                    report["records"].append(rec)
                    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
                    print(json.dumps(rec),flush=True)
    report["summary"] = {name:{"failures":sum(not r["valid"] for r in report["records"] if r["method"]==name),
        **{key:float(np.mean([r[key] for r in report["records"] if r["method"]==name])) for key in
           ("raw_mse","nominal_adjusted","residual_s","flops","cap_aware_score")}}
        for name in modules}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    main()
