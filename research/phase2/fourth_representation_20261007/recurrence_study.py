"""Small self-contained propagation, consistency, repair, and missing-third study."""
import argparse,hashlib,json,time,warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.concurrency import apply_thread_limit
from whestbench.generation import sample_mlp
from common import ROOT,load,measure,crossrisk
from math_reference import quadratic_matrix,sym_error,full_transport,planes
from oracle_study import NETWORKS,CHECK
import kernels

def consistency(states):
    out=[]
    for layer,s in enumerate(states):
        c=s['cov'];core=s['core'];parts=s['parts']
        rec=dict(layer=layer,cov_min_eigenvalue=float(np.linalg.eigvalsh(c)[0]),finite=bool(np.isfinite(c).all()),cov_symmetry=float(np.max(abs(c-c.T))))
        if core is not None:
            eigen=[];marg=[];sym=[]
            for block,k in zip(parts,core):
                cc=c[np.ix_(block,block)];eigen.append(float(np.linalg.eigvalsh(quadratic_matrix(k,cc))[0]))
                ids=np.arange(len(block));marg.extend(k[ids,ids,ids,ids]+2*np.diag(cc)**2);sym.append(sym_error(k))
            rec.update(S_min_eigenvalue=min(eigen),S_negative_blocks=int(np.sum(np.array(eigen)<-1e-7)),S_violation_sum=float(np.maximum(-np.array(eigen),0).sum()),marginal_min_slack=float(min(marg)),tensor_symmetry_error=max(sym),repair_alpha_max=0 if s['alpha'] is None else float(np.max(s['alpha'])))
        out.append(rec)
    return out

def offline_third(data):
    """Only generic core generation changes; all repeated-index anchors held fixed."""
    w=data['weights'].astype(np.float64);out=[]
    for source in (0,7,14):
        if source==14:continue # a subsequent actual layer is required
        ix=CHECK.index(source);jx=CHECK.index(source+1);ww=w[source+1];nextw=w[source+2]
        for b in (4,8,16):
            rule=json.loads((ROOT/'partition_selection.json').read_text())['selected_rule'];parts=data[f'parts_{b}_{rule}']
            errors={'complete_T':[],'repeated_only_T':[]}
            for stream in range(4):
                t=data['thirds'][stream,ix];n=len(t);i,j,k=np.ogrid[:n,:n,:n]
                repeated=np.where((i==j)|(i==k)|(j==k),t,0)
                kin=full_transport(data['fourths'][stream,ix],ww)
                c=ww.T@data['covs'][stream,source]@ww;m=data['means'][stream,source]@ww
                v=np.maximum(np.diag(c),1e-12);s=np.sqrt(v);a=m/s
                from oracle_study import ndtr
                phi=np.exp(-a*a/2)/np.sqrt(2*np.pi);gain=ndtr(a);second=phi/s;third=-a*phi/v
                for name,tt in (('complete_T',t),('repeated_only_T',repeated)):
                    tt=np.einsum('ijk,ia->ajk',tt,ww,optimize=True);tt=np.einsum('ajk,jb->abk',tt,ww,optimize=True);tt=np.einsum('abk,kc->abc',tt,ww,optimize=True)
                    local=np.asarray([tt[np.ix_(p,p,p)] for p in parts]);lc=np.asarray([c[np.ix_(p,p)] for p in parts]);lk=np.asarray([kin[np.ix_(p,p,p,p)] for p in parts])
                    with flops.BudgetContext(flop_budget=2**41,quiet=True):
                        generated=np.asarray(kernels.nonlinear_core(fnp.asarray(lk),fnp.asarray(lc),fnp.asarray(local),fnp.asarray(gain[parts]),fnp.asarray(second[parts]),fnp.asarray(third[parts])))
                    # Hold repeated-index entries at the same oracle post-nonlinearity values.
                    truth=data['fourths'][stream,jx];qm,rm=__import__('math_reference').pattern_masks(b)
                    represented=np.zeros_like(truth)
                    for p,core in zip(parts,generated):
                        actual=truth[np.ix_(p,p,p,p)]
                        represented[np.ix_(p,p,p,p)]=np.where(qm|rm,actual,core)
                    # The target here is the true BLOCK projection; cross-block loss excluded.
                    target=np.zeros_like(truth)
                    for p in parts:target[np.ix_(p,p,p,p)]=truth[np.ix_(p,p,p,p)]
                    delta=planes(represented-target,nextw)
                    errors[name].append(np.concatenate([x.ravel() for x in delta]))
            out.append(dict(source_layer=source,block_size=b,cross_stream_risk={k:crossrisk(v) for k,v in errors.items()},note='Offline oracle anchors held identical. Tests lost distinct third entries from source; not a submission or replacement for the CP-history algorithm.'))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='recurrence_results.json');parser.add_argument('--reuse-third',action='store_true');args=parser.parse_args()
    output=ROOT/args.output;assert not output.exists()
    rule=json.loads((ROOT/'partition_selection.json').read_text())['selected_rule']
    report=dict(partition=rule,records=[],offline_third=[],source_sha256={name:hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest() for name in ('baseline_diagnostic','exact_q_control','qr','block8','qblock8','qrblock8')})
    configs=[('baseline_diagnostic',None,False),('exact_q_control',None,False),('qr',None,False)]+[(name,b,False) for name in ('block8','qblock8','qrblock8') for b in (4,8,16)]+[('block8',8,True)]
    for n,seed in NETWORKS:
        data=np.load(ROOT/f'reference_n{n}_seed{seed}.npz');mlp=sample_mlp(n,16,fnp.random.default_rng(seed),seed=seed)
        predictions={};covariances={}
        for name,b,repair in configs:
            est=load(name).Estimator()
            if b is not None:est.block_size=b
            if name!='baseline_diagnostic':est.partition_rule=rule
            est.repair=repair
            label=name+(f'_b{b}' if b else '')+('_repair' if repair else '')
            started=time.perf_counter()
            try:
                pred,cost,states=measure(est,mlp,True)
                errors=pred[None]-data['means'];meanrisk=[crossrisk(errors[:,i]) for i in range(16)]
                crisk=[crossrisk(np.asarray(s['cov'])[None]-data['covs'][:,i]) for i,s in enumerate(states)]
                rec=dict(n=n,seed=seed,method=label,status='run',cost=cost,mean_cross_stream_risk=meanrisk,cov_cross_stream_risk=crisk,finite=bool(np.isfinite(pred).all()),consistency=consistency(states))
                predictions[label]=pred;covariances[label]=np.asarray([s['cov'] for s in states])
            except Exception as e:
                rec=dict(n=n,seed=seed,method=label,status='failed',exception=repr(e),wall_s=time.perf_counter()-started)
            report['records'].append(rec);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:rec[k] for k in ('n','seed','method','status')}|dict(final_risk=rec.get('mean_cross_stream_risk',[None])[-1])),flush=True)
        np.savez_compressed(ROOT/f'{output.stem}_n{n}_seed{seed}.npz',**predictions,**{'cov_'+k:v for k,v in covariances.items()})
        if not args.reuse_third:report['offline_third'].append(dict(n=n,seed=seed,records=offline_third(data)))
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    baseline=np.mean([r['mean_cross_stream_risk'][-1] for r in report['records'] if r['method']=='baseline_diagnostic'])
    scores={}
    for r in report['records']:
        if r['status']=='run':scores.setdefault(r['method'],[]).append(r['mean_cross_stream_risk'][-1])
    report['summary']=dict(baseline_final_risk=baseline,methods={k:dict(final_risk=float(np.mean(v)),improvement_fraction=float(1-np.mean(v)/baseline)) for k,v in scores.items()})
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
