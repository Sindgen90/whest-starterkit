"""Offline exploratory covariance diagnostics on public ID 0, fixed checkpoints.

References and source copies are never packaged. Adds only state observations
to production arithmetic; executes the same metered operations and budget.
"""
import hashlib,importlib.util,json,time,warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from whestbench import MLP
from whestbench.concurrency import apply_thread_limit
from common import ROOT,measure,risk_interval,crossrisk
from math_reference import quadratic_matrix,sym_error
CHECK=(0,7,14)
NAMES=('baseline_snapshot','qr','block8','block16','qblock8','qblock16','qrblock8','selective_block8')

def reference(weights):
    path=ROOT/'large_cov_reference_id0.npz'
    if path.exists():return np.load(path)
    means=[];covs=[]
    for stream in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([20261007,0,stream,1927]));sums=np.zeros((16,1024));second=np.zeros((3,1024,1024))
        for start in range(0,8192,1024):
            x=rng.standard_normal((1024,1024),dtype=np.float32)
            for layer,w in enumerate(weights):
                x=np.maximum(x@w,0);sums[layer]+=x.sum(0,dtype=np.float64)
                if layer in CHECK:
                    y=x.astype(np.float64);second[CHECK.index(layer)]+=y.T@y
        mu=sums/8192
        c=np.asarray([(second[i]-8192*np.outer(mu[layer],mu[layer]))/8191 for i,layer in enumerate(CHECK)])
        means.append(mu);covs.append(c.astype(np.float32));print(json.dumps(dict(stage='large_cov_reference',stream=stream)),flush=True)
    np.savez_compressed(path,means=means,covs=covs,checkpoints=CHECK,weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
    return np.load(path)

def observed_estimator(name):
    src=ROOT/name/'estimator.py';text=src.read_text();folder=ROOT/'offline_observers'/name;folder.mkdir(parents=True,exist_ok=True);path=folder/'estimator.py'
    if name=='baseline_snapshot':
        text=text.replace('        self.trace = []','        self.trace = []\n        self.states = []')
        needle='            cov = flops.as_symmetric(cov,symmetry=(0,1))'
        text=text.replace(needle,needle+'\n            if layer in (0,7,14):self.states.append(dict(layer=layer,mu=mu,cov=cov,q=fourth,r=None,core=None,parts=None))')
    else:
        needle='            if self.collect_diagnostics:\n                self.states.append('
        text=text.replace(needle,'            if layer in (0,7,14):\n                self.states.append(',1)
        text=text.replace('dict(mu=mu,cov=cov,q=fourth,r=rnew,core=core,parts=parts,pre=pre,d3=d3,d4=d4,alpha=alpha)','dict(layer=layer,mu=mu,cov=cov,q=fourth,r=rnew,core=core,parts=parts)')
    assert not path.exists();path.write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('observe_'+name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.Estimator(),hashlib.sha256(src.read_bytes()).hexdigest(),hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    path=ROOT/'large_covariance_results.json';assert not path.exists()
    row=Dataset.from_file('../tmp/phase2_mini_first8.arrow').with_format('numpy')[0]
    weights=np.asarray(row['weights'],dtype=np.float32);mlp=MLP(1024,16,list(fnp.asarray(weights)),seed=int(row.get('seed',0)))
    ref=reference(weights);report=dict(cohort='Exploratory public ID 0 only, checkpoints 0,7,14; independent 4x8192 reference inputs. Final mean ranking uses baked targets on IDs 0,1 separately. Observations do not enter inference.',records=[])
    for name in NAMES:
        est,sha,observer_sha=observed_estimator(name);started=time.perf_counter();exception=None
        try:pred,cost,states=measure(est,mlp,False);status='run'
        except Exception as e:
            status='failed';exception=repr(e);cost=None;pred=None
            states=[{k:np.asarray(v) if v is not None else None for k,v in s.items()} for s in est.states]
        diagnostics=[];covs={}
        for s in states:
            layer=int(s['layer']);c=s['cov'];errors=c[None]-ref['covs'][:,CHECK.index(layer)]
            d=dict(layer=layer,cov_cross_stream_risk=crossrisk(errors),cov_risk_interval=risk_interval(errors),cov_min_eigenvalue=float(np.linalg.eigvalsh(c)[0]),cov_symmetry=float(np.max(abs(c-c.T))))
            d.update(marginal_min_slack=float(np.min(np.diag(s['q'])+2*np.diag(c)**2)),q_symmetry_error=float(np.max(abs(s['q']-s['q'].T))),qr_diagonal_difference=None if s['r'] is None else float(np.max(abs(np.diag(s['q'])-np.diag(s['r'])))))
            if s['core'] is not None:
                eig=[];slack=[];sym=[]
                for p,k in zip(s['parts'],s['core']):
                    cc=c[np.ix_(p,p)];eig.append(float(np.linalg.eigvalsh(quadratic_matrix(k,cc))[0]));ids=np.arange(len(p));slack.extend(k[ids,ids,ids,ids]+2*np.diag(cc)**2);sym.append(sym_error(k))
                d.update(S_min_eigenvalue=min(eig),S_negative_blocks=int(np.sum(np.array(eig)<-1e-7)),S_violation_sum=float(np.maximum(-np.array(eig),0).sum()),marginal_min_slack=float(min(slack)),tensor_symmetry_error=max(sym))
            diagnostics.append(d);covs[f'cov_layer{layer}']=c
        rec=dict(method=name,status=status,exception=exception,source_sha256=sha,observer_sha256=observer_sha,cost=cost,diagnostics=diagnostics,wall_including_offline_checks_s=time.perf_counter()-started)
        report['records'].append(rec);path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        np.savez_compressed(ROOT/f'large_cov_states_{name}_id0.npz',**covs)
        print(json.dumps(dict(method=name,status=status,layers=[d['layer'] for d in diagnostics])),flush=True)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
