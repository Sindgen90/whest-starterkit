"""Use existing small-network full-third reference only; no new sampling."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def risk(e):
    e=np.array(e);n=len(e);return float(np.mean((e.sum(0)**2-(e*e).sum(0))/(n*(n-1))))
def main():
    out=R/'birth_diagnostic.json';assert not out.exists();records=[]
    for n,seed in ((16,82071),(16,82072),(32,82081),(32,82082)):
        d=np.load(R.parent/f'fourth_representation_20261007/reference_n{n}_seed{seed}.npz');w=d['weights'].astype(float);c=w[0].T@w[0];v=np.diag(c);w2=1/np.sqrt(2*np.pi*v);i,j,k=np.indices((n,n,n));distinct=(i!=j)&(i!=k)&(j!=k)
        tree=.25*(w2[:,None,None]*c[:,:,None]*c[:,None,:]+w2[None,:,None]*c[:,:,None]*c[None,:,:]+w2[None,None,:]*c[:,None,:]*c[None,:,:])
        true=d['thirds'][:,0];t=tree*distinct;ww=w[1];oracle_contractions=np.einsum('sijk,ia,ja,ka->sa',true*distinct,ww,ww,ww,optimize=True);estimate=np.einsum('ijk,ia,ja,ka->a',t,ww,ww,ww,optimize=True)
        records.append(dict(n=n,seed=seed,distinct_birth_risk_zero=risk(-true[:,distinct]),distinct_birth_risk_gaussian_tree=risk(tree[distinct][None,:]-true[:,distinct]),next_marginal_distinct_contraction_risk_zero=risk(-oracle_contractions),next_marginal_distinct_contraction_risk_tree=risk(estimate[None,:]-oracle_contractions)))
    report={'note':'First ReLU, exactly Gaussian incoming law. Existing four independent 32768-sample streams; cross-stream risk removes leading MC variance. Tree is the leading off-diagonal covariance term, not an exact full ReLU cumulant. These are small networks, not evidence of a width1024 adjusted-score gain.','records':records,'summary':{key:float(np.mean([r[key] for r in records])) for key in records[0] if key not in ('n','seed')}};out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
