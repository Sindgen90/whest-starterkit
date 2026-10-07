"""Offline independent reference streams and common-state representation tests."""
import argparse,hashlib,json,time,warnings,math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
def ndtr(a):return np.fromiter((.5*(1+math.erf(float(x)/math.sqrt(2))) for x in np.ravel(a)),dtype=np.float64).reshape(np.shape(a))
from whestbench.concurrency import apply_thread_limit
from whestbench.generation import sample_mlp
from common import ROOT,load,measure,crossrisk,interval,risk_interval
from math_reference import kstatistics,projection,planes,slices,full_transport
import kernels
CHECK=(0,1,7,8,14,15)
NETWORKS=((16,82071),(16,82072),(32,82081),(32,82082))

def reference(n,seed):
    path=ROOT/f'reference_n{n}_seed{seed}.npz'
    if path.exists():return np.load(path)
    mlp=sample_mlp(n,16,fnp.random.default_rng(seed),seed=seed)
    w=np.asarray(mlp.weights,dtype=np.float32)
    pred,cost,states=measure(load('baseline_diagnostic').Estimator(),mlp,True)
    parts={}
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        for b in (4,8,16):
            for rule in ('contiguous','random','correlation'):
                parts[f'parts_{b}_{rule}']=np.asarray(kernels.partition(fnp.asarray(states[0]['cov']),b,rule,seed))
    means=[];covs=[];thirds=[];fourths=[]
    for stream in range(4):
        started=time.perf_counter()
        rng=np.random.default_rng(np.random.SeedSequence([20261007,seed,stream,913]))
        x=rng.standard_normal((32768,n),dtype=np.float32)
        sm=[];sc=[];st=[];sk=[]
        for layer,ww in enumerate(w):
            x=np.maximum(x@ww,0)
            mu=x.mean(0,dtype=np.float64);y=x.astype(np.float64)-mu
            sm.append(mu);sc.append(y.T@y/(len(y)-1))
            if layer in CHECK:
                _,_,t,k=kstatistics(x.astype(np.float64));st.append(t);sk.append(k)
        means.append(sm);covs.append(sc);thirds.append(st);fourths.append(sk)
        print(json.dumps(dict(stage='reference',n=n,seed=seed,stream=stream,wall_s=time.perf_counter()-started)),flush=True)
    np.savez_compressed(path,weights=w,means=means,covs=covs,thirds=thirds,fourths=fourths,baseline=pred,baseline_first_cov=states[0]['cov'],**parts)
    return np.load(path)

def one_step(m,c,d3,p3,d4,q,r):
    v=np.maximum(np.diag(c),1e-12);s=np.sqrt(v);a=m/s
    phi=np.exp(-a*a/2)/np.sqrt(2*np.pi);p=ndtr(a)
    raw=[p,m*p+s*phi];raw.append(m*raw[1]+v*raw[0])
    for k in (3,4):raw.append(m*raw[k-1]+(k-1)*v*raw[k-2])
    f1=[None,p,phi/s,-a*phi/v,(a*a-1)*phi/(v*s)]
    f2=[None,2*raw[1],2*p,2*f1[2],2*f1[3]]
    f3=[None,3*raw[2],6*raw[1],6*p,6*f1[2]]
    f4=[None,None,None,24*raw[1],24*p]
    fs=[None,f1,f2,f3,f4]
    corr=[]
    for k,bound in zip(range(1,5),.25*np.array([s,v,v*s,v*v])):
        corr.append(raw[k]+np.clip(d3*fs[k][3]/6+d4*fs[k][4]/24,-bound,bound))
    mu=corr[0];rho=np.clip(c/np.outer(s,s),-.999999,.999999)
    half=.5*(a[:,None]**2+a[None,:]**2);ab=np.outer(a,a)
    integ=np.zeros_like(c)
    for node in (.21132486540518713,.7886751345948129):
        z=rho*node;den=1-z*z
        integ+=.5*(1-node)*np.exp(-(half-z*ab)/den)/(2*np.pi*np.sqrt(den))
    cov=c*np.outer(p,p)+np.outer(s,s)*rho*rho*integ
    pc=p3*c;dc=d3[:,None]*c
    for kernel,coef,(i,j) in zip((q,p3,p3.T,pc,pc.T,p3*p3.T,dc,dc.T),(.25,.5,.5,.5,.5,.25,1/6,1/6),((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))):
        cov+=coef*kernel*np.outer(f1[i],f1[j])
    roff=r.copy();np.fill_diagonal(roff,0)
    cov+=(roff*np.outer(f1[3],f1[1])+roff.T*np.outer(f1[1],f1[3]))/6
    cov=(cov+cov.T)/2;np.fill_diagonal(cov,np.maximum(corr[1]-mu*mu,1e-12))
    return mu,cov,raw,fs

def fourth_probes(k,w):
    n=len(w);out=[]
    for j in range(8):
        a,b,c,d=((3*j+1)%n,(5*j+2)%n,(7*j+3)%n,(11*j+4)%n)
        left=np.outer(w[:,a],w[:,b]).ravel();right=np.outer(w[:,c],w[:,d]).ravel()
        out.append(left@k.reshape((n*n,n*n))@right)
    return np.asarray(out)

def study():
    p=argparse.ArgumentParser();p.add_argument('--output',default='oracle_results.json');args=p.parse_args()
    output=ROOT/args.output;assert not output.exists()
    report=dict(protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),streams=4,samples_per_stream=32768,records=[],uncertainty='Linear cross-stream risks unbiased for linear projections of unbiased k-statistics. t intervals across four streams for squared plug-in losses are descriptive; nonlinear one-step errors remain plug-in estimates, not unbiased risks.')
    rules={r:[] for r in ('contiguous','random','correlation')}
    for n,seed in NETWORKS:
        data=reference(n,seed);w=data['weights'].astype(np.float64)
        for layer in (0,7,14):
            ww=w[layer+1];ix=CHECK.index(layer);jx=CHECK.index(layer+1)
            configs=[('A',None,None),('B',None,None),('C',None,None),('G',None,None)]+[(family,b,rule) for family in ('D','E','F') for b in (4,8,16) for rule in rules]
            truth=[];sources=[]
            for stream in range(4):
                k=data['fourths'][stream,ix];t=data['thirds'][stream,ix]
                c=ww.T@data['covs'][stream,layer]@ww;m=data['means'][stream,layer]@ww
                tt=np.einsum('ijk,ia->ajk',t,ww,optimize=True)
                tt=np.einsum('ajk,jb->abk',tt,ww,optimize=True)
                tt=np.einsum('abk,kc->abc',tt,ww,optimize=True)
                idx=np.arange(n);d3=tt[idx,idx,idx];p3=tt[idx[:,None],idx[:,None],idx[None,:]]
                targets=planes(k,ww)
                truth.append((*targets,fourth_probes(k,ww),data['means'][stream,layer+1],data['covs'][stream,layer+1]))
                sources.append((k,c,m,d3,p3))
            for family,b,rule in configs:
                errors=[[] for _ in range(6)];frob=[];plugloss=[]
                for stream,(k,c,m,d3,p3) in enumerate(sources):
                    parts=None if b is None else data[f'parts_{b}_{rule}']
                    projected=k if family=='G' else projection(k,{'A':'Q','B':'Q','C':'QR'}.get(family,family),parts)
                    if family=='A':
                        q,_=slices(k);s=ww*ww;gram=s.T@q@s
                        d4=3*np.diag(gram)-2*(s*s).T@np.diag(q);qp=gram.copy();np.fill_diagonal(qp,d4);rp=np.zeros_like(q)
                        contractions=fourth_probes(projected,ww)
                    else:d4,qp,rp=planes(projected,ww);contractions=fourth_probes(projected,ww)
                    mu,cc,_,_=one_step(m,c,d3,p3,d4,qp,rp)
                    estimates=(d4,qp,rp,contractions,mu,cc)
                    for i,(p,target) in enumerate(zip(estimates,truth[stream])):errors[i].append(p-target)
                    frob.append(float(np.mean((projected-k)**2)))
                    plugloss.append([float(np.mean(e[-1]**2)) for e in errors])
                names=('d4','Q','R','actual_weight_probes','one_step_mean','one_step_cov')
                rec=dict(n=n,seed=seed,source_layer=layer,family=family,block_size=b,partition=rule,
                    cross_stream_risk={name:crossrisk(e) for name,e in zip(names,errors)},
                    cross_stream_risk_intervals={name:risk_interval(e) for name,e in zip(names,errors)},
                    stream_plugin_loss_intervals={name:interval(np.asarray(plugloss)[:,i]) for i,name in enumerate(names)},tensor_plugin_loss=interval(frob))
                denom=[float(np.mean(np.mean([t[i] for t in truth],axis=0)**2))+1e-30 for i in range(3)]
                rec['normalized_transport_risk']=float(np.mean([rec['cross_stream_risk'][names[i]]/denom[i] for i in range(3)]))
                report['records'].append(rec)
                if family in ('E','F'):rules[rule].append(rec['normalized_transport_risk'])
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(dict(stage='oracle',n=n,seed=seed,layer=layer,records=len(report['records']))),flush=True)
    means={r:float(np.mean(v)) for r,v in rules.items()};best=min(means.values())
    selected=next(r for r in ('contiguous','random','correlation') if means[r]<=best+.01*max(abs(best),1e-12))
    result=dict(selected_rule=selected,normalized_transport_risk=means,criterion='Frozen PROTOCOL.md, averaged E/F d4/Q/R actual-weight cross-stream risks; 1% tie preference contiguous/random/correlation.',no_oracle_clusters=True)
    selection=ROOT/'partition_selection.json'
    if selection.exists():
        frozen=json.loads(selection.read_text());assert frozen['selected_rule']==result['selected_rule']
        np.testing.assert_allclose([frozen['normalized_transport_risk'][r] for r in rules],[result['normalized_transport_risk'][r] for r in rules],rtol=1e-12,atol=1e-12)
    else:selection.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');study()
