"""Fixed paired ablations. Timings here are diagnostics, never execution gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from whestbench.scoring import materialise_predictions
from whestbench.budget import score_multiplier

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'cumulant_audit_20261005'


def load(name):
    source = ROOT/'joint_full_runtime/estimator.py' if name=='baseline' else AUDIT/name/'estimator.py'
    spec = importlib.util.spec_from_file_location('audit_'+name,source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--names',nargs='+',default=['baseline','third_memory16','fourth_cross1','fourth_response'])
    p.add_argument('--shard',type=Path,default=Path('../tmp/phase2_mini_first8.arrow'))
    p.add_argument('--indices',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Never overwrite results.'
    loaded = {name:load(name) for name in a.names}
    hashes = {name:hashlib.sha256(src.read_bytes()).hexdigest() for name,(mod,src) in loaded.items()}
    report = {'shard':str(a.shard),'source_sha256':hashes,'records':[],
              'protocol':'Identical fixed networks; no fitting; four threads; actual current score multiplier. In-process runtime does not confer eligibility.'}
    data = Dataset.from_file(str(a.shard)).with_format('numpy')
    with threadpool_limits(4), warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for index in a.indices:
            row = data[index]
            weights = fnp.asarray(row['weights'],dtype=fnp.float32)
            mlp = MLP(weights.shape[1],len(weights),list(weights),seed=int(row.get('seed',0)))
            truth = np.asarray(row['all_layer_means'],dtype=np.float64)
            predictions = {}
            for name,(mod,src) in loaded.items():
                assert hashlib.sha256(src.read_bytes()).hexdigest()==hashes[name]
                est = mod.Estimator()
                started = time.perf_counter()
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
                    pred = est.predict(mlp,2**41)
                    arr = materialise_predictions(pred,depth=mlp.depth,width=mlp.width)
                    del pred
                wall = time.perf_counter()-started
                arr = np.asarray(arr,dtype=np.float64)
                assert arr.shape==truth.shape
                finite = bool(np.isfinite(arr).all())
                mse = float(np.mean((arr[-1]-truth[-1])**2))
                summary = budget.summary_dict()
                # Phase 2 has lambda=0: use the evaluator's unchanged multiplier.
                multiplier = score_multiplier(budget.flops_used,2**41,failed=False)
                rec = {'mlp_id':int(row['mlp_id']),'method':name,'raw_mse':mse,
                       'nominal_adjusted':mse*multiplier,'multiplier':multiplier,
                       'flops':budget.flops_used,'wall_s':wall,
                       'residual_s':summary['residual_wall_time_s'],'finite':finite,
                       'per_layer_mse':np.mean((arr-truth)**2,axis=1).tolist()}
                report['records'].append(rec)
                predictions[name] = arr
                a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in rec.items() if k!='per_layer_mse'}),flush=True)
            np.savez_compressed(a.output.with_name(a.output.stem+f'_mlp{int(row["mlp_id"])}.npz'),truth=truth,**predictions)
    report['summary'] = {name:{key:float(np.mean([r[key] for r in report['records'] if r['method']==name]))
                              for key in ('raw_mse','nominal_adjusted','flops','wall_s','residual_s')}
                         for name in a.names}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
