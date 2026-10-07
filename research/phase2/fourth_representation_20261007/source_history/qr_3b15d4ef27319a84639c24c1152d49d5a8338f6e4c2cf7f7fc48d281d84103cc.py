"""Experimental joint moment closure; no oracle targets or sampled moments.

Finite-memory third-cumulant factors and projected paired fourth cumulants.
Gaussian Cov(h,h), Cov(h^2,h), Cov(h^2,h^2) use Price-integral anchors.
Non-Gaussian response terms build on our local Puffi-inspired reconstruction;
this is not the published A716 algorithm. No fitted correction by default.
"""
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
def responses(mean, variance):
    """Gaussian M0..M4 and only the response derivatives consumed by the state."""
    # predict has already floored the incoming variance.
    sigma = fnp.sqrt(variance)
    r = mean / sigma
    phi = flops.stats.norm.pdf(r).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(r).astype(fnp.float32)
    moments = [cdf, mean * cdf + sigma * phi]
    moments.append(mean*moments[1] + variance*moments[0])
    for k in range(3, 5):
        moments.append(mean * moments[k-1] + (k-1) * variance * moments[k-2])
    f1 = [None, cdf, phi / sigma, -r * phi / variance,
          (r*r - 1) * phi / (variance*sigma)]
    f2 = [None, 2*moments[1], 2*cdf, 2*f1[2], 2*f1[3]]
    f3 = [None, None, None, 6*cdf, 6*f1[2]]
    f4 = [None, None, None, 24*moments[1], 24*cdf]
    return moments, [None, f1, f2, f3, f4], sigma, r, phi

def third_slices(a, b):
    """T=sum sym(a_r,b_r,b_r): return T[i,i,i] and P[i,j]=T[i,i,j]."""
    ab = a*b
    b2 = b*b
    diagonal = fnp.sum(ab*b, axis=1)
    repeated = (b2 @ a.T + 2 * (ab @ b.T)) / 3
    return diagonal, repeated


def final_mean(mean, variance, d3, d4):
    """Only the first ReLU moment is consumed at the terminal layer."""
    sigma = fnp.sqrt(variance)
    a = mean / sigma
    phi = flops.stats.norm.pdf(a).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
    gaussian = mean*cdf + sigma*phi
    f3 = -a*phi/variance
    f4 = (a*a-1)*phi/(variance*sigma)
    delta = f3*(d3/6) + f4*(d4/24)
    bound = .25*sigma
    return gaussian + fnp.clip(delta, -bound, bound)
_QUAD_NODES01 = (
    0.019855071751231912,
    0.10166676129318664,
    0.2372337950418355,
    0.4082826787521751,
    0.591717321247825,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_QUAD_WEIGHTS01 = (
    0.050614268145188185,
    0.11119051722668709,
    0.15685332293894372,
    0.18134189168918102,
    0.18134189168918102,
    0.15685332293894372,
    0.11119051722668709,
    0.050614268145188185,
)

QUADS = {8:tuple(zip(_QUAD_NODES01,_QUAD_WEIGHTS01)),
         2:((.21132486540518713,.5),(.7886751345948129,.5)),
         4:((.06943184420297371,.17392742256872692),(.33000947820757187,.32607257743127305),
            (.6699905217924281,.32607257743127305),(.9305681557970262,.17392742256872692))}


def quadrature_tables(order):
    one = fnp.ones((1,1),dtype=fnp.float32)
    return tuple(fnp.stack([one*value for value in values]) for values in
                 ([x for x,w in QUADS[order]],[w*(1-x) for x,w in QUADS[order]],
                  [w*(1-x)**2 for x,w in QUADS[order]]))


def gaussian_pairs(cov,moments,sigma,a,phi,need_higher=True,order=8,tables=None):
    """Gaussian covariance identities evaluated with Gauss-Legendre quadrature."""
    product = fnp.outer(sigma,sigma)
    rho = fnp.clip(cov/fnp.maximum(product,1e-30),-.999999,.999999)
    half_pair = .5*(a[:,None]*a[:,None]+a[None,:]*a[None,:])
    ab = fnp.outer(a,a)
    if tables is not None:
        nodes,weights11,weights22 = tables
        t = rho[None,:,:]*nodes
        denom = fnp.maximum(1.-t*t,1e-12)
        root = fnp.sqrt(denom)
        dens = fnp.exp(-(half_pair[None,:,:]-t*ab[None,:,:])/denom)/(6.283185307179586*root)
        integral11 = fnp.sum(weights11*dens,axis=0)
        if need_higher:
            conditional = flops.stats.norm.cdf((a[None,:,None]-t*a[None,None,:])/root).astype(fnp.float32)
            integral21 = fnp.sum(weights11*conditional,axis=0)
            integral22 = fnp.sum(weights22*dens,axis=0)
    else:
        integral11 = fnp.zeros(cov.shape,dtype=fnp.float32)
        if need_higher:
            integral21 = fnp.zeros(cov.shape,dtype=fnp.float32)
            integral22 = fnp.zeros(cov.shape,dtype=fnp.float32)
        for node,weight in QUADS[order]:
            t = rho*node
            denom = fnp.maximum(1.-t*t,1e-12)
            root = fnp.sqrt(denom)
            dens = fnp.exp(-(half_pair-t*ab)/denom)/(6.283185307179586*root)
            integral11 = integral11+weight*(1-node)*dens
            if need_higher:
                conditional = flops.stats.norm.cdf((a[:,None]-t*a[None,:])/root).astype(fnp.float32)
                integral21 = integral21+weight*(1-node)*conditional
                integral22 = integral22+weight*(1-node)**2*dens
    cdf,mean = moments[:2]
    gain_pair = fnp.outer(cdf,cdf)
    rho2 = rho*rho
    c11 = cov*gain_pair+product*rho2*integral11
    fnp.fill_diagonal(c11,fnp.maximum(moments[2]-mean*mean,1e-12))
    if not need_higher:
        return c11,None,None
    cov2 = cov*cov
    c21 = 2*cov*fnp.outer(mean,cdf)+2*cov2*integral21*(phi/sigma)[None,:]
    c22 = 4*cov*fnp.outer(mean,mean)+2*cov2*gain_pair+2*(product*product)*(rho2*rho)*integral22
    fnp.fill_diagonal(c21,moments[3]-moments[2]*mean)
    fnp.fill_diagonal(c22,moments[4]-moments[2]*moments[2])
    return c11,c21,c22


def nong_terms(cov,pair3,pair4,d3,f,need_higher=True,coefficients=None):
    """Batched response contractions; Gaussian terms are supplied separately."""
    pc = pair3*cov
    dc = d3[:,None]*cov
    # Apply response coefficients to O(n) vectors, not O(n^2) kernels.
    kernels = fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T))
    if coefficients is None:
        coefficients = fnp.asarray((.25,.5,.5,.5,.5,.25,1/6,1/6),dtype=fnp.float32)[:,None]
    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))
    left1 = fnp.stack([f[1][a] for a,b in orders])*coefficients
    right1 = fnp.stack([f[1][b] for a,b in orders])
    if not need_higher:
        return fnp.einsum('ki,kij,kj->ij',left1,kernels,right1),None,None
    left2 = fnp.stack([f[2][a] for a,b in orders])*coefficients
    right2 = fnp.stack([f[2][b] for a,b in orders])
    return tuple(fnp.einsum('ki,kij,kj->ij',left,kernels,right) for left,right in
                 ((left1,right1),(left2,right1),(left2,right2)))



"""Metered fourth-order kernels. No sampling, fitting or external reference data."""
import flopscope as flops
import flopscope.numpy as fnp

def global_planes(q,r,w,complete=False):
    s=w*w;d=fnp.diag(q)
    gram=s.T@q@s
    d4=3*fnp.diag(gram)-2*((s*s).T@d)
    pair=gram
    if complete:
        off=fnp.copy(q);fnp.fill_diagonal(off,0)
        pair=pair+2*fnp.einsum('ij,ia,ib,ja,jb->ab',off,w,w,w,w)
    cube=s*w
    # Exact R plane of the paired projection; all equal is counted once.
    rout=3*(w*(q@s)).T@w-2*cube.T@(d[:,None]*w)
    if r is not None:
        roff=fnp.copy(r);fnp.fill_diagonal(roff,0)
        rw=roff@w
        c=s.T@(w*rw)
        pair=pair+2*(c+c.T)
        rout=rout+cube.T@rw+3*(s*rw).T@w
        d4=d4+4*fnp.sum(cube*rw,axis=0)
    fnp.fill_diagonal(pair,d4);fnp.fill_diagonal(rout,d4)
    return d4,pair,rout

def partition(cov,b,rule,seed):
    n=cov.shape[0]
    if n%b:raise ValueError('Block size must divide width.')
    if rule=='contiguous':return fnp.arange(n,dtype=fnp.int64).reshape((-1,b))
    if rule=='random':return fnp.random.default_rng(20261007+int(seed or 0)).permutation(n).reshape((-1,b))
    if rule!='correlation':raise ValueError(rule)
    remaining=list(range(n));blocks=[];v=fnp.maximum(fnp.diag(cov),1e-30)
    while remaining:
        pivot=remaining[0];idx=fnp.asarray(remaining,dtype=fnp.int64)
        score=fnp.abs(cov[pivot,idx])/fnp.sqrt(v[pivot]*v[idx])
        # Stable sorting makes ties favor small indices. Include the pivot explicitly.
        positions=fnp.argsort(-score,kind='stable')
        chosen=[pivot]
        for k in positions:
            neuron=remaining[int(k)]
            if neuron!=pivot:chosen.append(neuron)
            if len(chosen)==b:break
        blocks.append(chosen);take=set(chosen);remaining=[i for i in remaining if i not in take]
    return fnp.asarray(blocks,dtype=fnp.int64)

def local_plane(plane,parts):return plane[parts[:,:,None],parts[:,None,:]]

def pattern_masks(b):
    idx=fnp.arange(b,dtype=fnp.int64)
    i=idx[:,None,None,None];j=idx[None,:,None,None]
    k=idx[None,None,:,None];l=idx[None,None,None,:]
    q=((i==j)&(k==l))|((i==k)&(j==l))|((i==l)&(j==k))
    r=((i==j)&(j==k))|((i==j)&(j==l))|((i==k)&(k==l))|((j==k)&(k==l))
    return q,r,((i==j)&(k==l)),((i==j)&(j==k)),((i==j)&(j==l)),((i==k)&(k==l))

def projected_core(q,r,masks):
    qm,rm,p1,t1,t2,t3=masks
    qt=fnp.where(p1[None],q[:,:,None,:,None],q[:,:,:,None,None])
    if r is None:return fnp.where(qm[None],qt,0)
    rt=fnp.where(t1[None],r[:,:,None,None,:],fnp.where(t2[None],r[:,:,None,:,None],fnp.where(t3[None],r[:,:,:,None,None],fnp.swapaxes(r,1,2)[:,:,:,None,None])))
    return fnp.where(qm[None],qt,fnp.where(rm[None],rt,0))

def anchor_core(core,q,r,masks):
    projected=projected_core(q,r,masks)
    return fnp.where((masks[0]|masks[1])[None],projected,core)

def symmetrize(core):
    v=(core+fnp.swapaxes(core,1,2))*.5
    v=(v+fnp.swapaxes(v,3,4))*.5
    v=(v+fnp.transpose(v,(0,3,4,1,2)))*.5
    return (v+fnp.swapaxes(v,2,3)+fnp.swapaxes(v,2,4))/3

def block_planes(core,parts,w):
    """Exact contractions; global covariance is never blocked."""
    wb=w[parts,:] # source block,i,all output
    blocks,b,n=wb.shape
    pairmap=(wb[:,:,None,:]*wb[:,None,:,:]).reshape((blocks,b*b,n))
    middle=core.reshape((blocks,b*b,b*b))@pairmap
    pair=fnp.sum(fnp.swapaxes(pairmap,1,2)@middle,axis=0)
    v=fnp.sum(middle.reshape((blocks,b,b,n))*wb[:,None,:,:],axis=2)
    r=fnp.sum(fnp.swapaxes(v,1,2)@wb,axis=0)
    return fnp.diag(pair),pair,r

def block_to_blocks(core,parts,w):
    """All source/target blocks interact; at most O(n^2*b^2) intermediates."""
    wb=w[parts[:,:,None,None],parts[None,None,:,:]] # source,i,target,a
    wb=fnp.transpose(wb,(0,2,1,3))
    v=fnp.einsum('sijkl,stia->stajkl',core,wb)
    v=fnp.einsum('stajkl,stjb->stabkl',v,wb)
    v=fnp.einsum('stabkl,stkc->stabcl',v,wb)
    v=fnp.einsum('stabcl,stld->stabcd',v,wb)
    return fnp.sum(v,axis=0)

def global_to_blocks(q,r,parts,w):
    """Exact global Q(+R) projection restricted to target blocks."""
    wb=fnp.transpose(w[:,parts],(1,0,2)) # target,input,a
    targets,n,b=wb.shape
    pairs=wb[:,:,:,None]*wb[:,:,None,:]
    p=pairs.reshape((targets,n,b*b))
    middle=q@p
    g=fnp.swapaxes(p,1,2)@middle
    g=g.reshape((targets,b,b,b,b))
    d=fnp.diag(q)
    diagonal=(fnp.swapaxes(p,1,2)@(d[None,:,None]*p)).reshape((targets,b,b,b,b))
    core=g+fnp.swapaxes(g,2,3)+fnp.swapaxes(g,2,4)-2*diagonal
    if r is not None:
        roff=fnp.copy(r);fnp.fill_diagonal(roff,0)
        rw=roff@wb
        triple=(pairs[:,:,:,:,None]*wb[:,:,None,None,:]).reshape((targets,n,b*b*b))
        term=(fnp.swapaxes(triple,1,2)@rw).reshape((targets,b,b,b,b))
        core=core+term+fnp.swapaxes(term,1,4)+fnp.swapaxes(term,2,4)+fnp.swapaxes(term,3,4)
    return core

def third_core(a,b,parts):
    aa=a[parts,:];bb=b[parts,:]
    blocks,width,rank=aa.shape
    pairs=(bb[:,:,None,:]*bb[:,None,:,:]).reshape((blocks,width*width,rank))
    t=(aa@fnp.swapaxes(pairs,1,2)).reshape((blocks,width,width,width))
    return (t+fnp.swapaxes(t,1,2)+fnp.swapaxes(t,1,3))/3

def nonlinear_core(incoming,cov,third,gain,second,third_response):
    """Connected Gaussian trees + linear K gain + leading covariance*third."""
    nb,b=gain.shape
    def vector(v,i):
        shape=[nb,1,1,1,1];shape[i+1]=b
        return v.reshape(tuple(shape))
    def pair(i,j):
        if i>j:i,j=j,i
        shape=[nb,1,1,1,1];shape[i+1]=shape[j+1]=b
        return cov.reshape(tuple(shape))
    def triple(i,j,k):
        ids=sorted((i,j,k));shape=[nb,1,1,1,1]
        for p in ids:shape[p+1]=b
        return third.reshape(tuple(shape))
    gains=[vector(gain,i) for i in range(4)]
    v=incoming*gains[0]*gains[1]*gains[2]*gains[3]
    for root in range(4):
        others=[i for i in range(4) if i!=root]
        term=vector(third_response,root)
        for j in others:term=term*gains[j]*pair(root,j)
        v=v+term
        for j in others:
            k,l=[p for p in others if p!=j]
            v=v+vector(second,root)*gains[j]*gains[k]*gains[l]*pair(root,j)*triple(root,k,l)
    for i in range(4):
        for j in range(i+1,4):
            k,l=[p for p in range(4) if p not in (i,j)]
            pref=vector(second,i)*vector(second,j)*gains[k]*gains[l]*pair(i,j)
            v=v+pref*(pair(i,k)*pair(j,l)+pair(i,l)*pair(j,k))
    return symmetrize(v)

def repair_core(core,cov):
    """PSD repair of quadratic-feature covariance, never PSD projection of K."""
    nb,b=cov.shape[:2]
    ij=[(i,j) for i in range(b) for j in range(i,b)]
    i=fnp.asarray([x for x,y in ij],dtype=fnp.int64);j=fnp.asarray([y for x,y in ij],dtype=fnp.int64)
    s=core[:,i[:,None],j[:,None],i[None,:],j[None,:]]+cov[:,i[:,None],i[None,:]]*cov[:,j[:,None],j[None,:]]+cov[:,i[:,None],j[None,:]]*cov[:,j[:,None],i[None,:]]
    eig=fnp.linalg.eigvalsh(s)
    minimum=eig[:,0]
    scale=fnp.maximum(1,fnp.max(fnp.diagonal(s,axis1=1,axis2=2),axis=1))
    alpha=fnp.where(minimum < -1e-10*scale,-minimum+1e-10*scale,0)
    eye=fnp.eye(b,dtype=fnp.float32)
    jtensor=eye[:,:,None,None]*eye[None,None,:,:]+eye[:,None,:,None]*eye[None,:,None,:]+eye[:,None,None,:]*eye[None,:,:,None]
    return core+alpha[:,None,None,None,None]*jtensor[None],alpha

"""Source fragment inlined after the frozen baseline helpers and metered kernels."""
def gaussian_extended(cov,moments,sigma,a,phi,tables):
    g11,g21,g22=gaussian_pairs(cov,moments,sigma,a,phi,tables=tables,order=2)
    product=fnp.outer(sigma,sigma)
    rho=fnp.clip(cov/fnp.maximum(product,1e-30),-.999999,.999999)
    nodes,weights,_=tables;t=rho[None]*nodes
    den=fnp.maximum(1-t*t,1e-12);root=fnp.sqrt(den)
    half=.5*(a[:,None]**2+a[None,:]**2)
    dens=fnp.exp(-(half[None]-t*fnp.outer(a,a)[None])/den)/(6.283185307179586*root)
    conditional=flops.stats.norm.cdf((a[None,:,None]-t*a[None,None,:])/root).astype(fnp.float32)
    first=fnp.sum(weights*den*dens,axis=0)*(sigma[:,None]/sigma[None,:])
    cm=sigma[None,:,None]*(a[None,:,None]-t*a[None,None,:])
    second=fnp.sum(weights*cm*conditional,axis=0)*(phi/sigma)[None,:]
    g31=3*cov*fnp.outer(moments[2],moments[0])+6*cov*cov*(first+second)
    fnp.fill_diagonal(g31,moments[4]-moments[3]*moments[1])
    return g11,g21,g22,g31

def nong_extended(cov,pair3,pair4,pair31,d3,f):
    pc=pair3*cov;dc=d3[:,None]*cov
    kernels=fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T))
    coef=fnp.asarray((.25,.5,.5,.5,.5,.25,1/6,1/6),dtype=fnp.float32)[:,None]
    orders=((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))
    left={k:fnp.stack([f[k][a] for a,b in orders])*coef for k in (1,2,3)}
    right={k:fnp.stack([f[k][b] for a,b in orders]) for k in (1,2)}
    roff=fnp.copy(pair31);fnp.fill_diagonal(roff,0)
    result=[]
    for x,y in ((1,1),(2,1),(2,2),(3,1)):
        v=fnp.einsum('ki,kij,kj->ij',left[x],kernels,right[y])
        v=v+(roff*fnp.outer(f[x][3],f[y][1])+roff.T*fnp.outer(f[x][1],f[y][3]))/6
        result.append(v)
    return result

class Estimator(BaseEstimator):
    family='QR'
    block_size=8
    partition_rule='correlation'
    complete_pair=False
    repair=False
    collect_diagnostics=False
    def predict(self,mlp,budget):
        n=mlp.width;nb=n//self.block_size
        blocked=self.family in ('D','E','F')
        globalq=self.family!='D';globalr=self.family in ('QR','F')
        mu=fnp.zeros(n,dtype=fnp.float32)
        cov=flops.as_symmetric(fnp.eye(n,dtype=fnp.float32),symmetry=(0,1))
        fourth=fnp.zeros((n,n),dtype=fnp.float32);rstate=None
        sources=[];rows=[];self.states=[];self.trace=[]
        parts=masks=core=None
        zero=fnp.zeros(n,dtype=fnp.float32)
        zplane=fnp.zeros((n,n),dtype=fnp.float32)
        tables=quadrature_tables(2)
        for layer,w in enumerate(mlp.weights):
            m=mu@w
            pre=fnp.einsum('ij,ia,jb->ab',cov,w,w)
            v=fnp.maximum(fnp.diag(pre),1e-12)
            moments,f,sigma,a,phi=responses(m,v)
            f[3][1]=3*moments[2];f[3][2]=6*moments[1]
            d3=d4=zero;pair3=pair4=pair31=zplane;incoming=None;aa=bb=None
            if layer:
                # Identical third-order affine transport, factorization and history rule.
                sources=[(born,w.T@aa,w.T if bb is None else w.T@bb) for born,aa,bb in sources]
                _,aa,bb=sources[0]
                d3,pair3=third_slices(aa,bb)
                if globalq:
                    d4,pair4,pair31=global_planes(fourth,rstate if globalr else None,w,self.complete_pair)
                    if self.family=='Q':pair31=zplane
                if blocked:
                    residual=core
                    if globalq:
                        residual=core-projected_core(local_plane(fourth,parts),local_plane(rstate,parts) if globalr else None,masks)
                    bd,bq,br=block_planes(residual,parts,w)
                    d4=d4+bd;pair4=pair4+bq;pair31=pair31+br
                    if layer<len(mlp.weights)-1 or self.collect_diagnostics:
                        incoming=block_to_blocks(residual,parts,w)
                        if globalq:incoming=incoming+global_to_blocks(fourth,rstate if globalr else None,parts,w)
                delta=fnp.stack([f[k][3] for k in range(1,5)])*(d3/6)[None]+fnp.stack([f[k][4] for k in range(1,5)])*(d4/24)[None]
                bound=.25*fnp.stack((sigma,v,v*sigma,v*v))
                corrected_array=fnp.stack(moments[1:])+fnp.clip(delta,-bound,bound)
                corrected=[moments[0]]+[corrected_array[k] for k in range(4)]
            else:corrected=moments
            mu=corrected[1];rows.append(mu)
            if layer==len(mlp.weights)-1 and not self.collect_diagnostics:break
            g11,g21,g22,g31=gaussian_extended(pre,moments,sigma,a,phi,tables)
            if layer:
                e11,e21,e22,e31=nong_extended(pre,pair3,pair4,pair31,d3,f)
                cov=(g11+e11);cov=(cov+cov.T)*.5
                fnp.fill_diagonal(cov,fnp.maximum(corrected[2]-mu*mu,1e-12))
                cov21=g21+e21;cov22=g22+e22;cov31=g31+e31
            else:cov=g11;cov21=g21;cov22=g22;cov31=g31
            post3=corrected[3]-3*mu*corrected[2]+2*mu*mu*mu
            twice_mu=2*mu
            repeated=cov21-twice_mu[:,None]*cov
            fnp.fill_diagonal(repeated,post3)
            post4=corrected[4]-4*mu*corrected[3]-3*corrected[2]*corrected[2]+12*mu*mu*corrected[2]-6*fnp.power(mu,4)
            fourth=cov22-twice_mu[None,:]*cov21-twice_mu[:,None]*cov21.T+fnp.outer(twice_mu,twice_mu)*cov-2*cov*cov
            fourth=(fourth+fourth.T)*.5;fnp.fill_diagonal(fourth,post4)
            rnew=cov31-3*mu[:,None]*cov21+3*(mu*mu)[:,None]*cov-3*fnp.diag(cov)[:,None]*cov
            fnp.fill_diagonal(rnew,post4)
            alpha=None
            if blocked:
                if parts is None:
                    parts=partition(cov,self.block_size,self.partition_rule,mlp.seed)
                    masks=pattern_masks(self.block_size)
                lc=local_plane(pre,parts)
                lt=fnp.zeros((nb,self.block_size,self.block_size,self.block_size),dtype=fnp.float32) if aa is None else third_core(aa,bb,parts)
                if incoming is None:incoming=fnp.zeros((nb,)+(self.block_size,)*4,dtype=fnp.float32)
                core=nonlinear_core(incoming,lc,lt,f[1][1][parts],f[1][2][parts],f[1][3][parts])
                core=anchor_core(core,local_plane(fourth,parts),local_plane(rnew,parts),masks)
                if self.repair:core,alpha=repair_core(core,local_plane(cov,parts))
            rstate=rnew if globalr else None
            fourth=flops.as_symmetric(fourth,symmetry=(0,1))
            # Exactly the baseline's one-generation third source residual reinjection.
            gain=f[1][1]
            sources=[(born,gain[:,None]*x,gain[:,None]*y) for born,x,y in sources if layer-born<1]
            if sources:
                old3,oldp=third_slices(sources[0][1],sources[0][2])
                post3=post3-old3;repeated=repeated-oldp
            anew=repeated.T;fnp.multiply(anew,3,out=anew);fnp.fill_diagonal(anew,post3)
            sources.append((layer,anew,None))
            cov=flops.as_symmetric(cov,symmetry=(0,1))
            if self.collect_diagnostics:
                self.states.append(dict(mu=mu,cov=cov,q=fourth,r=rnew,core=core,parts=parts,pre=pre,d3=d3,d4=d4,alpha=alpha))
        return fnp.stack(rows)
