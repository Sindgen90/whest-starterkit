"""Build standalone conditional closure and independent random-direction control."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base=(ROOT.parent/'joint_full_runtime/estimator.py').read_text()
base=base.replace('class Estimator(BaseEstimator):','class JointClosure:')
base=base.replace('    def predict(self,mlp,budget):','    def propagate(self,mlp,budget,initial_mu,initial_cov):')
base=base.replace('        mu = fnp.zeros(n,dtype=fnp.float32)','        mu = initial_mu')
base=base.replace('        cov = flops.as_symmetric(fnp.eye(n,dtype=fnp.float32),symmetry=(0,1))','        cov = flops.as_symmetric(initial_cov,symmetry=(0,1))')
base=base.replace("                v = fnp.maximum(fnp.einsum('ij,ia,ja->a',cov,w,w),1e-12)","                raw_v = fnp.einsum('ij,ia,ja->a',cov,w,w)\n                deterministic = raw_v<=1e-7*fnp.sum(w*w,axis=0) if layer==0 else raw_v==0\n                v = fnp.maximum(raw_v,1e-12)")
base=base.replace('                rows.append(final_mean(m,v,d3,d4))','                rows.append(fnp.where(deterministic,fnp.maximum(m,0),final_mean(m,v,d3,d4)))')
base=base.replace('            v = fnp.maximum(fnp.diag(pre),1e-12)','            raw_v = fnp.diag(pre)\n            deterministic = raw_v<=1e-7*fnp.sum(w*w,axis=0) if layer==0 else raw_v==0\n            v = fnp.maximum(raw_v,1e-12)')
needle='            d3 = zero\n            d4 = zero\n'
assert base.count(needle)==1
base=base.replace(needle,'''            # Singular Gaussian directions are deterministic, with zero central moments.
            value=fnp.maximum(m,0)
            moments=[fnp.where(deterministic,(m>0).astype(fnp.float32),moments[0])]+[fnp.where(deterministic,value**k,moments[k]) for k in range(1,len(moments))]
            phi=fnp.where(deterministic,0,phi)
            if rich:
                f[1][1]=moments[0]
            deterministic_pairs=deterministic[:,None]|deterministic[None,:]
            d3 = zero
            d4 = zero
''')
base=base.replace('            rows.append(mu)','            mu=fnp.where(deterministic,value,mu)\n            if rich:\n                corrected=[moments[0]]+[fnp.where(deterministic,value**k,corrected[k]) for k in range(1,5)]\n            rows.append(mu)')
base=base.replace('            if need_state:\n                cov21', '            cov=fnp.where(deterministic_pairs,0,cov)\n            if need_state:\n                cov21')
base=base.replace('                post3 = corrected[3]-3*mu*corrected[2]+2*mu*mu*mu','                cov21=fnp.where(deterministic_pairs,0,cov21)\n                cov22=fnp.where(deterministic_pairs,0,cov22)\n                post3 = corrected[3]-3*mu*corrected[2]+2*mu*mu*mu\n                post3=fnp.where(deterministic,0,post3)')
base=base.replace('                    fourth = (cov22', '                    post4=fnp.where(deterministic,0,post4)\n                    fourth = (cov22')
outer='''

def important_direction(mlp,rng):
    """Leading mode from budgeted stochastic VJPs, without a full Jacobian."""
    x=rng.standard_normal((8,mlp.width)).astype(fnp.float32)
    masks=[]
    for w in mlp.weights:
        z=x@w
        mask=(z>0).astype(fnp.float32)
        masks.append(mask)
        x=fnp.maximum(z,0)
    # Independent output probes at each independent input point.
    gradient=rng.standard_normal((8,4,mlp.width)).astype(fnp.float32)
    for layer in range(mlp.depth-1,-1,-1):
        gradient=gradient*masks[layer][:,None,:]
        gradient=(gradient.reshape((32,mlp.width))@mlp.weights[layer].T).reshape((8,4,mlp.width))
    gradient=gradient.reshape((32,mlp.width))
    values,vectors=fnp.linalg.eigh(gradient@gradient.T)
    u=gradient.T@vectors[:,-1]
    norm=fnp.sqrt(fnp.sum(u*u))
    fallback=rng.standard_normal(mlp.width).astype(fnp.float32)
    fallback=fallback/fnp.maximum(fnp.sqrt(fnp.sum(fallback*fallback)),1e-30)
    return fnp.where(norm>1e-20,u/fnp.maximum(norm,1e-30),fallback)


class Estimator(BaseEstimator):
    random_seed=20261211
    use_random_direction=False
    def predict(self,mlp,budget):
        rng=fnp.random.default_rng(self.random_seed+1009*int(mlp.seed or 0))
        if self.use_random_direction:
            u=rng.standard_normal(mlp.width).astype(fnp.float32)
            u=u/fnp.maximum(fnp.sqrt(fnp.sum(u*u)),1e-30)
        else:
            u=important_direction(mlp,rng)
        cov=flops.as_symmetric(fnp.eye(mlp.width,dtype=fnp.float32)-fnp.outer(u,u),symmetry=(0,1))
        rows=[]
        for node,weight in ((0.,2/3),(1.7320508075688772,1/6),(-1.7320508075688772,1/6)):
            rows.append(weight*JointClosure().propagate(mlp,budget,u*node,cov))
        result=fnp.sum(fnp.stack(rows),axis=0)
        w=mlp.weights[0]
        first=fnp.sqrt(fnp.sum(w*w,axis=0))/2.5066282746310002
        return fnp.concatenate((first[None,:],result[1:]),axis=0)
'''
for name,random in (('conditional',False),('random_control',True)):
    p=ROOT/name/'estimator.py';p.parent.mkdir();assert not p.exists()
    p.write_text(base+outer.replace('use_random_direction=False',f'use_random_direction={random}'))
print('Built conditional closure and random-direction control.')
