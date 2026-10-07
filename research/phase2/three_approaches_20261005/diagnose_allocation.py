"""Offline small-network distinct-third repairs; no oracle submission data."""
import json,hashlib,time,warnings
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
from whestbench.generation import sample_mlp
from experiment import ROOT,load,measure

def rank_correlation(x,y):
    def ranks(x):
        values,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
        starts=np.cumsum(counts)-counts
        return (starts+(counts-1)/2)[inverse]
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def sample(weights,seed,stream,purpose):
    rng=np.random.default_rng(np.random.SeedSequence([20261005,seed,stream,purpose]))
    means=np.zeros((8,16)); middle=[]
    for offset in range(0,32768,2048):
        x=rng.standard_normal((2048,16),dtype=np.float32)
        for l,w in enumerate(weights):
            x=np.maximum(x@w,0)
            means[l]+=x.sum(0,dtype=np.float64)
            if l==2 and purpose==1:middle.append(x.copy())
    return means/32768,np.concatenate(middle) if middle else None

def tensor(a,b):
    full=(np.einsum('i,j,k->ijk',a,b,b)+np.einsum('i,j,k->ijk',b,a,b)+np.einsum('i,j,k->ijk',b,b,a))/3
    idx=np.indices(full.shape);full[(idx[0]==idx[1])|(idx[0]==idx[2])|(idx[1]==idx[2])]=0
    return full

def main():
    out=ROOT/'allocation_diagnostic.json';assert not out.exists()
    diag,src=load('repair_diagnostic');allocation,apsrc=load('allocation')
    report={'protocol':'PROTOCOL.md: independent projection and assessment streams; n=16,d=8, individual discarded distinct CP columns at post-layer index 2. No rule fitted from references.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,apsrc)},'records':[]}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for seed in (20261061,20261062):
            mlp=sample_mlp(16,8,fnp.random.default_rng(seed),seed=seed);weights=np.asarray(mlp.weights,dtype=np.float32)
            captured={}
            def capture(layer,sources):
                if layer==2:
                    captured['a']=np.concatenate([np.asarray(a) for _,a,b in sources],axis=1).copy()
                    captured['b']=np.concatenate([np.asarray(b) for _,a,b in sources],axis=1).copy()
                return []
            est=diag.Estimator();est.repair_hook=capture
            base,counts=measure(est,mlp)
            projection=[sample(weights,seed,k,1) for k in range(4)]
            assessment=np.asarray([sample(weights,seed,k,2)[0] for k in range(4)])
            y=np.concatenate([s[1] for s in projection]).astype(np.float64);y-=y.mean(0)
            true=np.zeros((16,16,16))
            for offset in range(0,len(y),2048):
                z=y[offset:offset+2048]
                pairs=(z[:,:,None]*z[:,None,:]).reshape((len(z),256))
                true+=(pairs.T@z).reshape((16,16,16))
            true/=len(y)
            with flops.BudgetContext(flop_budget=2**41,quiet=True):
                probes,curv=allocation.output_probes(mlp)
                scores=np.asarray(allocation.distinct_scores(fnp.asarray(captured['a']),fnp.asarray(captured['b']),probes[2],curv))
            aa,bb=captured['a'],captured['b']
            norms=np.sqrt(np.sum(aa*aa,axis=0))*np.sum(bb*bb,axis=0)
            repairs=[];predictions=[]
            for r in range(aa.shape[1]):
                direction=tensor(aa[:,r].astype(float),bb[:,r].astype(float));norm2=np.sum(direction*direction)
                coef=float(np.sum(true*direction)/norm2) if norm2>1e-30 else 0.
                def repair(layer,sources,r=r,coef=coef):
                    if layer!=2:return []
                    a=fnp.concatenate([a for _,a,b in sources],axis=1)
                    b=fnp.concatenate([b for _,a,b in sources],axis=1)
                    return [(layer,a[:,r:r+1]*coef,b[:,r:r+1])]
                est=diag.Estimator();est.repair_hook=repair
                pred,cost=measure(est,mlp);predictions.append(pred)
                delta=np.mean((pred[None,-1]-assessment[:,-1])**2-(base[None,-1]-assessment[:,-1])**2,axis=1)
                mean=float(delta.mean());half=float(3.182446305284263*delta.std(ddof=1)/2)
                repairs.append({'column':r,'coefficient':coef,'norm_bound':float(norms[r]),'unit_sensitivity':float(scores[r]/max(norms[r],1e-30)),'sensitivity_rank_score':float(scores[r]),'offline_error_weighted_score':float(abs(coef)*scores[r]),'benefit':-mean,'delta_raw_mse':mean,'delta_95_sampling':[mean-half,mean+half],'mean_change_norm':float(np.linalg.norm(pred[-1]-base[-1])),'finite':bool(np.isfinite(pred).all())})
            benefit=np.asarray([r['benefit'] for r in repairs]);effect=np.asarray([r['mean_change_norm'] for r in repairs]);weighted=np.asarray([r['offline_error_weighted_score'] for r in repairs])
            correlations={name:rank_correlation(rank,benefit) for name,rank in [('norm',norms),('sensitivity',scores),('offline_error_weighted',weighted)]}
            rec={'weight_seed':seed,'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),'baseline_counts':counts,'rank_correlation_with_benefit':correlations,'weighted_correlation_with_effect_magnitude':rank_correlation(weighted,effect),'informative':correlations['sensitivity']>correlations['norm']+.2,'repairs':repairs}
            report['records'].append(rec);out.write_text(json.dumps(report,indent=2))
            np.savez_compressed(ROOT/f'allocation_diagnostic_seed{seed}.npz',weights=weights,projection_means=np.asarray([s[0] for s in projection]),assessment_means=assessment,true_third= true,old_a=aa,old_b=bb,baseline=base,individual_repairs=np.asarray(predictions))
            print(json.dumps({k:v for k,v in rec.items() if k!='repairs'}),flush=True)
    report['informative_on_both']=all(r['informative'] for r in report['records']);out.write_text(json.dumps(report,indent=2));print(json.dumps({'informative_on_both':report['informative_on_both']}),flush=True)
if __name__=='__main__':main()
