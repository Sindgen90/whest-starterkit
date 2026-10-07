"""Create diagnostic hooks and the frozen standalone allocation prototype."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=(ROOT.parent/'joint_full_runtime/estimator.py').read_text()
OLD='''                sources = [(born,gain[:,None]*aa,gain[:,None]*bb)
                           for born,aa,bb in sources if layer-born<self.history]
'''
assert BASE.count(OLD)==1
hook='''                transported = [(born,gain[:,None]*aa,gain[:,None]*bb)
                               for born,aa,bb in sources]
                sources = self.repair_hook(layer,transported)
'''
p=ROOT/'repair_diagnostic/estimator.py';p.parent.mkdir();assert not p.exists();p.write_text(BASE.replace(OLD,hook))
helpers='''
def output_probes(mlp):
    """Diagonal Gaussian future model; eight final-coordinate probes."""
    n=mlp.width
    mu=fnp.zeros(n,dtype=fnp.float32)
    var=fnp.ones(n,dtype=fnp.float32)
    gains=[]
    for w in mlp.weights:
        m=mu@w
        v=fnp.maximum((w*w).T@var,1e-12)
        moments,f,sigma,a,phi=responses(m,v)
        mu=moments[1]
        var=fnp.maximum(moments[2]-mu*mu,1e-12)
        gains.append(f[1][1])
    indices=fnp.asarray([int(i*(n-1)/7) for i in range(8)],dtype=fnp.int64)
    curvature=f[1][3][indices]/6
    q=fnp.eye(n,dtype=fnp.float32)[:,indices]
    probes=[None]*(mlp.depth-1)
    for layer in range(mlp.depth-1,0,-1):
        q=mlp.weights[layer]@q
        probes[layer-1]=q
        q=gains[layer-1][:,None]*q
    return probes,curvature

def distinct_scores(a,b,q,curvature):
    """Exact cubic contraction of each CP column minus repeated patterns."""
    ap=q.T@a
    bp=q.T@b
    q2=q*q
    b2=b*b
    ab=a*b
    b2p=q2.T@b2
    abp=q2.T@ab
    ab2p=(q2*q).T@(ab*b)
    delta=ap*(bp*bp-b2p)-2*abp*bp+2*ab2p
    weighted=curvature[:,None]*delta
    return fnp.sqrt(fnp.sum(weighted*weighted,axis=0))

'''
selection='''                sources = [(born,gain[:,None]*aa,gain[:,None]*bb)
                           for born,aa,bb in sources]
                if sources:
                    aa=fnp.concatenate([aa for born,aa,bb in sources],axis=1)
                    bb=fnp.concatenate([bb for born,aa,bb in sources],axis=1)
                    scores=distinct_scores(aa,bb,probes[layer],curvature)
                    selected=fnp.argsort(scores)[-min(16,aa.shape[1]):]
                    sources=[(layer,aa[:,selected],bb[:,selected])]
'''
text=BASE.replace('class Estimator(BaseEstimator):',helpers+'class Estimator(BaseEstimator):').replace('        n = mlp.width\n','        n = mlp.width\n        probes,curvature=output_probes(mlp)\n',1).replace(OLD,selection)
p=ROOT/'allocation/estimator.py';p.parent.mkdir();assert not p.exists();p.write_text(text)
print('Built diagnostic-only hooks and standalone allocation prototype.')
