"""Serial accuracy/cost screen; eligibility requires separate official subprocess gate."""
import argparse,hashlib,importlib.util,json,time,warnings,sys
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from whestbench import MLP,SetupContext
from whestbench.scoring import materialise_predictions
from whestbench.concurrency import apply_thread_limit
R=Path(__file__).resolve().parent
def load(name):
    p=R/name/'estimator.py';s=importlib.util.spec_from_file_location('test_'+name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--ids',nargs='+',type=int,default=[0,1]);p.add_argument('--data',default='../tmp/phase2_mini_first8.arrow');p.add_argument('--output',required=True);p.add_argument('--diagnostics',action='store_true');p.add_argument('--reuse',action='store_true');a=p.parse_args()
    path=R/a.output;assert not path.exists();data=Dataset.from_file(a.data).with_format('numpy')
    report={'source_sha256':{n:hashlib.sha256((R/n/'estimator.py').read_bytes()).hexdigest() for n in a.names},'records':[],'timing':'in-process only; no execution eligibility inferred','reuse_setup_once':a.reuse}
    estimators={};report['setup_wall_s']={}
    if a.reuse:
        for name in a.names:
            est=load(name).Estimator();started=time.perf_counter();est.setup(SetupContext(width=1024,depth=16,flop_budget=2**41,api_version='0.16.1',seed=20261071));report['setup_wall_s'][name]=time.perf_counter()-started;estimators[name]=est
    for idx in a.ids:
        row=data[idx];w=fnp.asarray(row['weights'],dtype=fnp.float32);mlp=MLP(w.shape[1],len(w),list(w),seed=int(row.get('seed',0)));truth=np.asarray(row['all_layer_means'],dtype=np.float64);arrays={'truth':truth}
        for name in a.names:
            started=time.perf_counter();est=estimators[name] if a.reuse else load(name).Estimator();est.collect_diagnostics=a.diagnostics
            try:
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as b:
                    pred=np.asarray(materialise_predictions(est.predict(mlp,2**41),depth=16,width=1024),dtype=np.float64)
                cost=b.summary_dict();mse=float(np.mean((pred[-1]-truth[-1])**2));f=int(b.flops_used)
                rec=dict(method=name,mlp_id=int(row['mlp_id']),finite=bool(np.isfinite(pred).all()),raw_mse=mse,nominal_adjusted=mse*max(.1,f/2**41),flops=f,wall_s=time.perf_counter()-started,residual_s=cost['residual_wall_time_s'],per_layer_mse=np.mean((pred-truth)**2,axis=1).tolist())
                arrays[name]=pred
                if a.diagnostics:
                    for k,s in enumerate(est.states):
                        for key,value in s.items():arrays[f'{name}_{k}_{key}']=np.asarray(value)
            except Exception as e:rec=dict(method=name,mlp_id=int(row['mlp_id']),exception=repr(e),wall_s=time.perf_counter()-started)
            report['records'].append(rec);path.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!='per_layer_mse'}),flush=True)
        np.savez_compressed(R/(path.stem+f'_id{idx}.npz'),**arrays)
    report['summary']={name:{k:float(np.mean([x[k] for x in report['records'] if x['method']==name and k in x])) for k in ('raw_mse','nominal_adjusted','flops','wall_s','residual_s') if any(x['method']==name and k in x for x in report['records'])} for name in a.names};path.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
