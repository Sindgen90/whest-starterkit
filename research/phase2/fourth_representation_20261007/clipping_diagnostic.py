"""Offline recentering/guard diagnostic; changes no estimator or selection."""
import json
import numpy as np
from common import ROOT,crossrisk,risk_interval
from oracle_study import NETWORKS,CHECK,ndtr
from math_reference import planes
from whestbench.concurrency import apply_thread_limit

def corrected(m,v,d3,d4):
    s=np.sqrt(v);a=m/s;phi=np.exp(-a*a/2)/np.sqrt(2*np.pi);cdf=ndtr(a)
    raw=[cdf,m*cdf+s*phi];raw.append(m*raw[1]+v*raw[0])
    for k in (3,4):raw.append(m*raw[k-1]+(k-1)*v*raw[k-2])
    third=np.asarray([-a*phi/v,2*phi/s,6*cdf,24*raw[1]])
    fourth=np.asarray([(a*a-1)*phi/(v*s),-2*a*phi/v,6*phi/s,24*cdf])
    delta=third*d3/6+fourth*d4/24;bound=.25*np.array([s,v,v*s,v*v])
    clipped=np.array(raw[1:])+np.clip(delta,-bound,bound)
    unclipped34=clipped.copy();unclipped34[2:]=np.array(raw[3:])+delta[2:]
    return clipped,unclipped34,np.abs(delta)>bound

def fourth_of(mom):
    mu,m2,m3,m4=mom
    return m4-4*mu*m3-3*m2*m2+12*mu*mu*m2-6*mu**4

def translation_example():
    out=[]
    for mean in (10.,20.):
        # Valid finite distribution entirely above zero: ReLU is the identity.
        x=mean+np.array([-.5,1.5]);p=np.array([.75,.25]);v=.75;t=.75;k=-.375
        clipped,unclipped,_=corrected(np.array([mean]),np.array([v]),np.array([t]),np.array([k]))
        out.append(dict(mean=mean,input_variance=v,input_third=t,true_input_and_output_fourth=k,clipped_fourth=float(fourth_of(clipped)[0]),unclipped34_fourth=float(fourth_of(unclipped)[0]),clipped_marginal_slack=float(fourth_of(clipped)[0]+2*v*v),support=x.tolist(),probabilities=p.tolist()))
    return out

def main():
    apply_thread_limit(4)
    path=ROOT/'clipping_diagnostic.json';assert not path.exists()
    report=dict(purpose='Post-hoc offline explanation, not a selected estimator. Hold corrected mean/M2 fixed and remove only raw-M3/M4 clipping in the complete-fourth oracle. No recurrence, inference or submission change; no new fitted coefficient.',translation_counterexample=translation_example(),records=[])
    for n,seed in NETWORKS:
        data=np.load(ROOT/f'reference_n{n}_seed{seed}.npz');w=data['weights'].astype(np.float64);idx=np.arange(n)
        for source in (0,7,14):
            ww=w[source+1];ix=CHECK.index(source);jx=CHECK.index(source+1);errors={'clipped':[],'unclipped34':[]};fractions=[]
            for stream in range(4):
                t=data['thirds'][stream,ix];tt=np.einsum('ijk,ia->ajk',t,ww,optimize=True);tt=np.einsum('ajk,jb->abk',tt,ww,optimize=True);tt=np.einsum('abk,kc->abc',tt,ww,optimize=True)
                d3=tt[idx,idx,idx];d4=planes(data['fourths'][stream,ix],ww)[0]
                m=data['means'][stream,source]@ww;c=ww.T@data['covs'][stream,source]@ww
                clipped,unclipped,mask=corrected(m,np.maximum(np.diag(c),1e-12),d3,d4)
                target=data['fourths'][stream,jx][idx,idx,idx,idx]
                errors['clipped'].append(fourth_of(clipped)-target);errors['unclipped34'].append(fourth_of(unclipped)-target)
                fractions.append(mask.mean(axis=1))
            report['records'].append(dict(n=n,seed=seed,source_layer=source,risks={k:crossrisk(v) for k,v in errors.items()},intervals={k:risk_interval(v) for k,v in errors.items()},raw_moment_clip_fractions=np.mean(fractions,axis=0).tolist()))
    report['summary']={k:float(np.mean([r['risks'][k] for r in report['records']])) for k in ('clipped','unclipped34')}
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');return report
if __name__=='__main__':print(json.dumps(main()['summary']))
