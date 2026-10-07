"""Offline paired quadrature-order control. Does not modify the submission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP

ROOT = Path(__file__).parent
SOURCE = ROOT/"venture_baseline"/"estimator.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=8)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("quadrature_control",SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = Dataset.from_file(str(args.arrow)).with_format("numpy")
    report = {"protocol":"Orders 8,16,32; fixed published calibration, float32, 4 threads; no fitting",
              "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "dataset":str(args.arrow),"records":[]}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    budget = 2**41
    original_nodes,original_weights = mod._QUAD_NODES01,mod._QUAD_WEIGHTS01
    with threadpool_limits(4):
        for i in range(min(args.limit,len(data))):
            row = data[i]
            ws = fnp.asarray(row["weights"],dtype=fnp.float32)
            mlp = MLP(ws.shape[1],len(ws),list(ws),seed=20260930)
            target = np.asarray(row["all_layer_means"],dtype=np.float64)
            baseline = None
            for order in (8,16,32):
                if order==8:
                    mod._QUAD_NODES01,mod._QUAD_WEIGHTS01 = original_nodes,original_weights
                else:
                    nodes,weights = np.polynomial.legendre.leggauss(order)
                    mod._QUAD_NODES01 = tuple(((nodes+1)/2).tolist())
                    mod._QUAD_WEIGHTS01 = tuple((weights/2).tolist())
                est = mod.Estimator()
                est.setup(SimpleNamespace(submission_dir=SOURCE.parent,seed=20260930))
                rec = {"mlp_id":int(row["mlp_id"]),"order":order}
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",UserWarning)
                        with flops.BudgetContext(flop_budget=budget,quiet=True) as ctx:
                            pred = est.predict(mlp,budget)
                    wall = time.perf_counter()-start
                    pred = np.asarray(pred,dtype=np.float64)
                    assert pred.shape==target.shape and np.isfinite(pred).all()
                    if order==8:
                        baseline = pred.copy()
                    residual = ctx.summary_dict()["residual_wall_time_s"]
                    mse = float(np.mean((pred[-1]-target[-1])**2))
                    score = mse*max(.1,ctx.flops_used/budget)
                    valid = wall<=120 and residual<=.4 and ctx.flops_used<=budget
                    rec.update(raw_mse=mse,nominal_adjusted=score,flops=ctx.flops_used,
                        utilization=ctx.flops_used/budget,wall_s=wall,residual_s=residual,valid=valid,
                        cap_aware_score=score if valid else float(np.mean(target[-1]**2)),
                        final_change_mse=float(np.mean((pred[-1]-baseline[-1])**2)),
                        max_final_change=float(np.max(np.abs(pred[-1]-baseline[-1]))))
                except Exception as exc:
                    rec.update(error=repr(exc),valid=False)
                report["records"].append(rec)
                args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps(rec),flush=True)
    report["summary"] = {}
    for order in (8,16,32):
        rows = [r for r in report["records"] if r["order"]==order]
        report["summary"][order] = {"failures":sum(not r["valid"] for r in rows),
            **{k:float(np.mean([r[k] for r in rows])) for k in
               ("raw_mse","nominal_adjusted","cap_aware_score","final_change_mse","utilization","residual_s")}}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2),flush=True)


if __name__=="__main__":
    main()
