"""One serial paired development screen. Execution eligibility is a separate gate."""
import argparse,hashlib,json,time,warnings
import numpy as np
import flopscope.numpy as fnp
from datasets import Dataset
from whestbench import MLP
from whestbench.concurrency import apply_thread_limit
from whestbench.budget import score_multiplier
from common import ROOT,load,measure

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['baseline_snapshot','qr','block8','block16','qblock8','qblock16','qrblock8','qrblock16']);p.add_argument('--output',default='full_width_results.json');a=p.parse_args()
    output=ROOT/a.output;assert not output.exists()
    data=Dataset.from_file('../tmp/phase2_mini_first8.arrow').with_format('numpy')
    report=dict(timing='In-process screen only; unchanged official subprocess gate decides eligibility.',cohort='Public baked 1e9-draw means, IDs 0,1. No fitting.',records=[],source_sha256={name:hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest() for name in a.names})
    for index in (0,1):
        row=data[index];w=fnp.asarray(row['weights'],dtype=fnp.float32);mlp=MLP(w.shape[1],len(w),list(w),seed=int(row.get('seed',0)))
        truth=np.asarray(row['all_layer_means'],dtype=np.float64);predictions={'truth':truth}
        for name in a.names:
            started=time.perf_counter()
            try:
                pred,cost,_=measure(load(name).Estimator(),mlp)
                finite=bool(np.isfinite(pred).all());mse=float(np.mean((pred[-1]-truth[-1])**2))
                multiplier=score_multiplier(cost['flops'],2**41,failed=False)
                rec=dict(mlp_id=int(row['mlp_id']),method=name,status='run',finite=finite,raw_mse=mse,nominal_adjusted=mse*multiplier,multiplier=multiplier,per_layer_mse=np.mean((pred-truth)**2,axis=1).tolist(),**cost)
                predictions[name]=pred
            except Exception as e:rec=dict(mlp_id=int(row['mlp_id']),method=name,status='failed',exception=repr(e),wall_s=time.perf_counter()-started)
            report['records'].append(rec);output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in rec.items() if k!='per_layer_mse'}),flush=True)
        np.savez_compressed(ROOT/(output.stem+f'_mlp{index}.npz'),**predictions)
    report['summary']={name:{key:float(np.mean([r[key] for r in report['records'] if r['method']==name and key in r])) for key in ('raw_mse','nominal_adjusted','flops','wall_s','residual_s') if any(r['method']==name and key in r for r in report['records'])} for name in a.names}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
