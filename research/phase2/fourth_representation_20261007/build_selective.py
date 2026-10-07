"""One prescribed source-layer schedule; no coefficient or schedule search."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base=(ROOT/'baseline_snapshot/estimator.py').read_text();helpers=base[:base.index('class Estimator(BaseEstimator):')]
kernel=(ROOT/'kernels.py').read_text();closure=(ROOT/'closure.py').read_text()
def replace(old,new):
    global closure
    assert old in closure,old[:80]
    closure=closure.replace(old,new,1)
replace("family='QR'","family='D'")
replace("partition_rule='contiguous'","partition_rule='correlation'")
replace('    repair=False','    repair=False\n    selective_sources=(0,7)')
replace('        for layer,w in enumerate(mlp.weights):','''        for layer,w in enumerate(mlp.weights):
            previous_active=layer>0 and layer-1 in self.selective_sources
            source_active=layer in self.selective_sources
            globalq=not previous_active;globalr=False;blocked=previous_active''')
replace('''                    d4,pair4,pair31=global_planes(fourth,rstate if globalr else None,w,self.complete_pair)
                    if self.family=='Q':pair31=zplane''','''                    square=w*w
                    pair4=fnp.einsum('ij,ia,jb->ab',fourth,square,square)
                    d4=3*fnp.diag(pair4)-2*((square*square).T@fnp.diag(fourth))
                    fnp.fill_diagonal(pair4,d4)
                    pair31=zplane''')
replace('''                delta=fnp.stack([f[k][3]''','''                if source_active and not previous_active:
                    incoming=global_to_blocks(fourth,None,parts,w)
                delta=fnp.stack([f[k][3]''')
replace('''                    if layer<len(mlp.weights)-1 or self.collect_diagnostics:''','''                    if source_active:''')
replace('''            g11,g21,g22,g31=gaussian_extended(pre,moments,sigma,a,phi,tables)''','''            if source_active:g11,g21,g22,g31=gaussian_extended(pre,moments,sigma,a,phi,tables)
            else:g11,g21,g22=gaussian_pairs(pre,moments,sigma,a,phi,tables=tables,order=2);g31=None''')
replace('''                e11,e21,e22,e31=nong_extended(pre,pair3,pair4,pair31,d3,f)''','''                if previous_active or source_active:e11,e21,e22,e31=nong_extended(pre,pair3,pair4,pair31,d3,f)
                else:e11,e21,e22=nong_terms(pre,pair3,pair4,d3,f);e31=None''')
replace('''                cov21=g21+e21;cov22=g22+e22;cov31=g31+e31''','''                cov21=g21+e21;cov22=g22+e22;cov31=g31+e31 if source_active else None''')
replace('''            rnew=cov31-3*mu[:,None]*cov21+3*(mu*mu)[:,None]*cov-3*fnp.diag(cov)[:,None]*cov
            fnp.fill_diagonal(rnew,post4)
            alpha=None
            if blocked:''','''            rnew=None
            if source_active:
                rnew=cov31-3*mu[:,None]*cov21+3*(mu*mu)[:,None]*cov-3*fnp.diag(cov)[:,None]*cov
                fnp.fill_diagonal(rnew,post4)
            alpha=None
            if source_active:''')
replace('''                if self.repair:core,alpha=repair_core(core,local_plane(cov,parts))
            rstate''','''                if self.repair:core,alpha=repair_core(core,local_plane(cov,parts))
            else:core=None
            rstate''')
# Same deterministic greedy pivot-neighbor rule, with batched index operations.
start=kernel.index('    remaining=list(range(n))');end=kernel.index('\ndef local_plane',start)
kernel=kernel[:start]+'''    idx=fnp.arange(n,dtype=fnp.int64);blocks=[];v=fnp.maximum(fnp.diag(cov),1e-30)
    while idx.shape[0]:
        pivot=int(idx[0])
        score=fnp.abs(cov[pivot,idx])/fnp.sqrt(v[pivot]*v[idx])
        score=fnp.where(idx==pivot,float('inf'),score)
        chosen=idx[fnp.argsort(-score,kind='stable')[:b]]
        blocks.append(chosen)
        keep=fnp.all(idx[:,None]!=chosen[None,:],axis=1)
        idx=idx[keep]
    return fnp.stack(blocks)
''' +kernel[end:]
folder=ROOT/'selective_block8';folder.mkdir(exist_ok=True);src=folder/'estimator.py'
if src.exists():
    old=src.read_bytes();(ROOT/'source_history'/('selective_block8_'+hashlib.sha256(old).hexdigest()+'.py')).write_bytes(old)
    oldselection=(ROOT/'selective_selection.json').read_bytes();(ROOT/'source_history'/('selective_selection_'+hashlib.sha256(oldselection).hexdigest()+'.json')).write_bytes(oldselection)
src.write_text(helpers+'\n'+kernel+'\n'+closure,encoding='utf-8')
record=dict(family='D',block_size=8,partition='correlation',source_layers=[0,7],selected_before_confirmation=True,
    justification='Frozen schedule in PROTOCOL.md. D,b8 is strongest small recurrence; all-layer full-width D,b8 exceeds the residual cap in screen. No selective coefficient/schedule sweep.',
    reversion='At a rich source the represented K is block-only. After consumption, recompute global Q from the current full covariance and marginal/bivariate closure; do not copy discarded cross-block K. At source 7 construct incoming core by restricting the current global Q projection.',
    source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
(ROOT/'selective_selection.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
