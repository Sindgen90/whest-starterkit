"""Oracle-input bivariate fourth-generation errors, separate from recurrence."""
import json,warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.concurrency import apply_thread_limit
from common import ROOT,load,crossrisk,risk_interval
from math_reference import projection,planes,slices
from oracle_study import NETWORKS,CHECK

def update(mod,m,c,d3,p3,d4,q,r):
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        m,c,d3,p3,d4,q,r=[fnp.asarray(x,dtype=fnp.float32) for x in (m,c,d3,p3,d4,q,r)]
        v=fnp.maximum(fnp.diag(c),1e-12);raw,f,s,a,phi=mod.responses(m,v)
        f[3][1]=3*raw[2];f[3][2]=6*raw[1]
        delta=fnp.stack([f[k][3] for k in range(1,5)])*(d3/6)[None]+fnp.stack([f[k][4] for k in range(1,5)])*(d4/24)[None]
        bound=.25*fnp.stack((s,v,v*s,v*v));corr=fnp.stack(raw[1:])+fnp.clip(delta,-bound,bound)
        mu=corr[0]
        gauss=mod.gaussian_extended(c,raw,s,a,phi,mod.quadrature_tables(2));nong=mod.nong_extended(c,p3,q,r,d3,f)
        cc=gauss[0]+nong[0];cc=(cc+cc.T)*.5;fnp.fill_diagonal(cc,fnp.maximum(corr[1]-mu*mu,1e-12))
        b21=gauss[1]+nong[1];b22=gauss[2]+nong[2];b31=gauss[3]+nong[3]
        twice=2*mu
        qpost=b22-twice[None]*b21-twice[:,None]*b21.T+fnp.outer(twice,twice)*cc-2*cc*cc
        qpost=(qpost+qpost.T)*.5
        rpost=b31-3*mu[:,None]*b21+3*(mu*mu)[:,None]*cc-3*fnp.diag(cc)[:,None]*cc
        dpost=corr[3]-4*mu*corr[2]-3*corr[1]*corr[1]+12*mu*mu*corr[1]-6*mu**4
        fnp.fill_diagonal(qpost,dpost);fnp.fill_diagonal(rpost,dpost)
        result=tuple(np.asarray(x) for x in (mu,cc,dpost,qpost,rpost))
    return result

def main():
    path=ROOT/'local_nonlinear_results.json';assert not path.exists();mod=load('qr')
    rule=json.loads((ROOT/'partition_selection.json').read_text())['selected_rule']
    report=dict(protocol='Identical oracle mu,C,T; only represented input fourth tensor changes. Four independent reference streams. Same bivariate/marginal closure as inference. Oracle inputs stay offline. Cross-stream risks/intervals remain nonlinear plug-in diagnostics.',records=[])
    for n,seed in NETWORKS:
        data=np.load(ROOT/f'reference_n{n}_seed{seed}.npz');w=data['weights'].astype(np.float64);ids=np.arange(n)
        for source in (0,7,14):
            ix=CHECK.index(source);jx=CHECK.index(source+1);ww=w[source+1]
            errors={name:[[] for _ in range(5)] for name in ('A','C','D8','E8','F8','G')}
            for stream in range(4):
                k=data['fourths'][stream,ix];t=data['thirds'][stream,ix]
                tt=np.einsum('ijk,ia->ajk',t,ww,optimize=True);tt=np.einsum('ajk,jb->abk',tt,ww,optimize=True);tt=np.einsum('abk,kc->abc',tt,ww,optimize=True)
                d3=tt[ids,ids,ids];p3=tt[ids[:,None],ids[:,None],ids[None,:]]
                m=data['means'][stream,source]@ww;c=ww.T@data['covs'][stream,source]@ww
                targetq,targetr=slices(data['fourths'][stream,jx])
                targets=(data['means'][stream,source+1],data['covs'][stream,source+1],np.diag(targetq),targetq,targetr)
                for name in errors:
                    if name=='A':
                        q,_=slices(k);s=ww*ww;qp=s.T@q@s;d4=3*np.diag(qp)-2*(s*s).T@np.diag(q);np.fill_diagonal(qp,d4);rp=np.zeros_like(qp)
                    else:
                        family={'C':'QR','D8':'D','E8':'E','F8':'F'}.get(name)
                        kk=k if name=='G' else projection(k,family,data[f'parts_8_{rule}'] if name.endswith('8') else None)
                        d4,qp,rp=planes(kk,ww)
                    estimates=update(mod,m,c,d3,p3,d4,qp,rp)
                    for i,(a,b) in enumerate(zip(estimates,targets)):errors[name][i].append(a-b)
            for name,e in errors.items():
                labels=('mean','cov','post_d4','post_Q','post_R')
                report['records'].append(dict(n=n,seed=seed,source_layer=source,family=name,risks={key:crossrisk(x) for key,x in zip(labels,e)},intervals={key:risk_interval(x) for key,x in zip(labels,e)}))
            path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(n=n,seed=seed,source_layer=source)),flush=True)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
