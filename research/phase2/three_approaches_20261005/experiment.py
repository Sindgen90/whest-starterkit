"""Append-only paired measurements; in-process times are descriptive only."""
import argparse,hashlib,importlib.util,json,time,warnings
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP
from whestbench.scoring import materialise_predictions
from whestbench.budget import score_multiplier

ROOT=Path(__file__).resolve().parent
def load(name):
    if name=='baseline': p=ROOT.parent/'joint_full_runtime/estimator.py'
    elif name=='norm_control': p=ROOT.parent/'cumulant_audit_20261005/third_memory16/estimator.py'
    else: p=ROOT/name/'estimator.py'
    spec=importlib.util.spec_from_file_location('three_'+name,p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,p

def measure(est,mlp):
    start=time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as b:
        p=est.predict(mlp,2**41)
        a=materialise_predictions(p,depth=mlp.depth,width=mlp.width)
        del p
    s=b.summary_dict()
    return np.asarray(a,dtype=np.float64),{'flops':b.flops_used,'wall_s':time.perf_counter()-start,'residual_s':s['residual_wall_time_s'],'multiplier':score_multiplier(b.flops_used,2**41,failed=False)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['pilot','screen'])
    parser.add_argument('--names',nargs='+',default=['baseline','surrogate'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[20261211])
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists()
    mods={name:load(name) for name in a.names}
    report={'mode':a.mode,'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,(_,p) in mods.items()},'records':[],'timing':'In-process descriptive only; unchanged official subprocess is execution gate.'}
    data=Dataset.from_file('../tmp/phase2_mini_first8.arrow').with_format('numpy')
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for index in (0,1):
            row=data[index];w=fnp.asarray(row['weights'],dtype=fnp.float32)
            mlp=MLP(w.shape[1],len(w),list(w),seed=int(row.get('seed',0)))
            truth=np.asarray(row['all_layer_means'],dtype=np.float64)
            predictions={'truth':truth}
            for name,(mod,source) in mods.items():
                if a.mode=='pilot' and name=='surrogate':
                    with flops.BudgetContext(flop_budget=2**41,quiet=True) as b:
                        d,e=mod.dictionary(mlp)
                        coef=mod.fit(mlp,fnp.random.default_rng(20261201+index),d)
                        x=fnp.random.default_rng(20261202+index).standard_normal((1024,mlp.width)).astype(fnp.float32)
                        values=mod.forward(x,mlp.weights)
                        residual=values-mod.features(x,d)@coef
                        residual=np.asarray(residual).reshape((1024,mlp.depth,mlp.width))
                        values=np.asarray(values).reshape(residual.shape)
                    # Offline pilot test does not enter inference or candidate cost.
                    v=np.var(residual[:,-1],axis=0,ddof=1);rawv=np.var(values[:,-1],axis=0,ddof=1)
                    rec={'mlp_id':int(row['mlp_id']),'method':name,'mean_final_residual_variance':float(v.mean()),'variance_ratio':float(v.mean()/rawv.mean()),'predicted_raw_mse_4096':float(v.mean()/4096),'pilot_flops_including_test':b.flops_used,'note':'Independent test inputs. Prediction excludes floating-point fitting error; actual inference FLOPs measured separately.'}
                    np.savez_compressed(a.output.with_name(a.output.stem+f'_residual_mlp{index}.npz'),residual_variance=v,raw_variance=rawv)
                    report['records'].append(rec);print(json.dumps(rec),flush=True)
                else:
                    seeds=a.seeds if name in ('surrogate','conditional','random_control') else [a.seeds[0]]
                    for seed in seeds:
                        est=mod.Estimator();est.random_seed=seed
                        pred,counts=measure(est,mlp)
                        finite=bool(np.isfinite(pred).all());mse=float(np.mean((pred[-1]-truth[-1])**2))
                        rec={'mlp_id':int(row['mlp_id']),'method':name,'random_seed':seed,'raw_mse':mse,'nominal_adjusted':mse*counts['multiplier'],'finite':finite,'per_layer_mse':np.mean((pred-truth)**2,axis=1).tolist(),**counts}
                        report['records'].append(rec);predictions[f'{name}_{seed}']=pred
                        print(json.dumps({k:v for k,v in rec.items() if k!='per_layer_mse'}),flush=True)
                a.output.write_text(json.dumps(report,indent=2))
            np.savez_compressed(a.output.with_name(a.output.stem+f'_mlp{index}.npz'),**predictions)
    report['summary']={name:{key:float(np.mean([r[key] for r in report['records'] if r['method']==name and key in r])) for key in ('raw_mse','nominal_adjusted','flops','wall_s','residual_s') if any(r['method']==name and key in r for r in report['records'])} for name in a.names}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':main()
