"""Paired check of frozen upload source on networks excluded from this scout."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from benchmark_runtime import load

ROOT = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--start',type=int,default=4)
p.add_argument('--limit',type=int,default=4)
p.add_argument('--candidate',default='joint_candidate')
args = p.parse_args()
modules = {name:load(name) for name in ('venture_runtime',args.candidate)}
report = {'protocol':'Frozen candidate. No fitting. Networks 4-7 and 12-15 excluded from current scout; previously exposed in earlier research. Four backend threads.',
          'source_sha256':{name:hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest() for name in modules},'records':[]}
with threadpool_limits(4),warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning)
    for path in ('../tmp/phase2_mini_first8.arrow','../tmp/phase2_mini_validation8.arrow'):
        data = Dataset.from_file(path).with_format('numpy')
        for i in range(args.start,min(args.start+args.limit,len(data))):
            row = data[i]
            weights = fnp.asarray(row['weights'],dtype=fnp.float32)
            mlp = MLP(weights.shape[1],len(weights),list(weights),seed=20261001)
            truth = np.asarray(row['all_layer_means'],dtype=np.float64)
            for name,module in modules.items():
                est = module.Estimator()
                est.setup(SimpleNamespace(submission_dir=ROOT/name,seed=20261001))
                start = time.perf_counter()
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
                    pred = np.asarray(est.predict(mlp,2**41),dtype=np.float64)
                wall = time.perf_counter()-start
                assert pred.shape==truth.shape and np.isfinite(pred).all()
                mse = float(np.mean((pred[-1]-truth[-1])**2))
                residual = budget.summary_dict()['residual_wall_time_s']
                valid = residual<=.4 and wall<=120 and budget.flops_used<=2**41
                rec = dict(method=name,mlp_id=int(row['mlp_id']),raw_mse=mse,nominal_adjusted=mse*max(.1,budget.flops_used/2**41),
                           flops=budget.flops_used,residual_s=residual,wall_s=wall,valid=valid,
                           cap_aware_score=mse*max(.1,budget.flops_used/2**41) if valid else float(np.mean(truth[-1]**2)))
                report['records'].append(rec)
                args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(rec),flush=True)
report['summary'] = {name:{'failures':sum(not r['valid'] for r in report['records'] if r['method']==name),
    **{key:float(np.mean([r[key] for r in report['records'] if r['method']==name])) for key in
       ('raw_mse','nominal_adjusted','cap_aware_score','residual_s','flops')}} for name in modules}
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['summary'],indent=2),flush=True)
