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
    """Gaussian positive-part moments M0..M4 and derivatives F[k][q], q=1..4."""
    sigma = fnp.sqrt(fnp.maximum(variance, 1e-12))
    r = mean / sigma
    phi = flops.stats.norm.pdf(r).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(r).astype(fnp.float32)
    moments = [cdf, mean * cdf + sigma * phi]
    for k in range(2, 5):
        moments.append(mean * moments[k-1] + (k-1) * variance * moments[k-2])
    f1 = [None, cdf, phi / sigma, -r * phi / variance,
          (r*r - 1) * phi / (variance*sigma)]
    f2 = [None, 2*moments[1], 2*cdf, 2*f1[2], 2*f1[3]]
    f3 = [None, 3*moments[2], 6*moments[1], 6*cdf, 6*f1[2]]
    f4 = [None, 4*moments[3], 12*moments[2], 24*moments[1], 24*cdf]
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


class Estimator(BaseEstimator):
    history = 1
    use_fourth = True
    collect_diagnostics = False
    rich_layers = 16
    quadrature_order = 2
    batch_quadrature = True

    def predict(self,mlp,budget):
        n = mlp.width
        mu = fnp.zeros(n,dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n,dtype=fnp.float32),symmetry=(0,1))
        fourth = fnp.zeros((n,n),dtype=fnp.float32)
        sources = []
        rows = []
        self.trace = []
        zero = fnp.zeros(n,dtype=fnp.float32)
        tables = quadrature_tables(self.quadrature_order) if self.batch_quadrature else None
        coefficients = fnp.asarray((.25,.5,.5,.5,.5,.25,1/6,1/6),dtype=fnp.float32)[:,None]
        for layer,w in enumerate(mlp.weights):
            m = mu@w
            rich = layer<self.rich_layers
            if layer==len(mlp.weights)-1 and not self.collect_diagnostics:
                # No later layer consumes covariance, P, Q, or raw moments 2..4.
                # Contract their diagonals directly instead of forming matrices.
                v = fnp.maximum(fnp.einsum('ij,ia,ja->a',cov,w,w),1e-12)
                d3 = d4 = zero
                if rich and layer>0:
                    terms = []
                    for _,aa,bb in sources:
                        aa = w.T@aa
                        bb = w.T if bb is None else w.T@bb
                        terms.append(fnp.sum((aa*bb)*bb,axis=1))
                    if terms:
                        d3 = terms[0] if len(terms)==1 else fnp.sum(fnp.stack(terms),axis=0)
                    if self.use_fourth:
                        square = w*w
                        diagonal = fnp.einsum('ij,ia,ja->a',fourth,square,square)
                        d4 = 3*diagonal-2*((square*square).T@fnp.diag(fourth))
                rows.append(final_mean(m,v,d3,d4))
                break
            pre = fnp.einsum('ij,ia,jb->ab',cov,w,w)
            v = fnp.maximum(fnp.diag(pre),1e-12)
            if rich:
                moments,f,sigma,a,phi = responses(m,v)
            else:
                sigma = fnp.sqrt(v)
                a = m/sigma
                phi = flops.stats.norm.pdf(a).astype(fnp.float32)
                cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
                moments = [cdf,m*cdf+sigma*phi,(m*m+v)*cdf+m*sigma*phi]
            d3 = zero
            d4 = zero
            if rich and layer>0:
                sources = [(born,w.T@aa,w.T if bb is None else w.T@bb) for born,aa,bb in sources]
                if sources:
                    if len(sources)==1:
                        _,aa,bb = sources[0]
                    else:
                        aa = fnp.concatenate([aa for born,aa,bb in sources],axis=1)
                        bb = fnp.concatenate([bb for born,aa,bb in sources],axis=1)
                    d3,pair3 = third_slices(aa,bb)
                else:
                    pair3 = fnp.zeros((n,n),dtype=fnp.float32)
                if self.use_fourth:
                    square = w*w
                    pair4 = fnp.einsum('ij,ia,jb->ab',fourth,square,square)
                    d4 = 3*fnp.diag(pair4)-2*((square*square).T@fnp.diag(fourth))
                    # Retain the actual projected Gram plane instead of a row-mean surrogate.
                    fnp.fill_diagonal(pair4,d4)
                else:
                    pair4 = fnp.zeros((n,n),dtype=fnp.float32)
                delta = (fnp.stack([f[k][3] for k in range(1,5)])*(d3/6)[None,:]
                         +fnp.stack([f[k][4] for k in range(1,5)])*(d4/24)[None,:])
                bound = .25*fnp.stack((sigma,v,v*sigma,v*v))
                corrected_array = fnp.stack(moments[1:])+fnp.clip(delta,-bound,bound)
                corrected = [moments[0]]+[corrected_array[k] for k in range(4)]
                mu = corrected[1]
            else:
                corrected = moments
                mu = moments[1]
            rows.append(mu)
            if self.collect_diagnostics:
                mass = (fnp.sum(fnp.abs(pre),axis=1)-v)/n
                self.trace.append((m,v,d3,d4,mu,mass))
            if layer==len(mlp.weights)-1:
                break
            need_state = rich and layer+1<self.rich_layers
            g11,g21,g22 = gaussian_pairs(pre,moments,sigma,a,phi,need_higher=need_state,order=self.quadrature_order,tables=tables)
            if rich and layer>0:
                e11,e21,e22 = nong_terms(pre,pair3,pair4,d3,f,need_higher=need_state,coefficients=coefficients)
                cov = g11+e11
                marginal_var = fnp.maximum(corrected[2]-mu*mu,1e-12)
                fnp.fill_diagonal(cov,marginal_var)
                cov = .5*(cov+cov.T)
            else:
                cov = g11
            if need_state:
                cov21 = g21+e21 if layer>0 else g21
                cov22 = g22+e22 if layer>0 else g22
                post3 = corrected[3]-3*mu*corrected[2]+2*mu*mu*mu
                repeated = cov21-2*mu[:,None]*cov
                fnp.fill_diagonal(repeated,post3)
                if self.use_fourth:
                    post4 = (corrected[4]-4*mu*corrected[3]-3*corrected[2]*corrected[2]
                             +12*mu*mu*corrected[2]-6*fnp.power(mu,4))
                    fourth = (cov22-2*mu[None,:]*cov21-2*mu[:,None]*cov21.T
                              +4*fnp.outer(mu,mu)*cov-2*cov*cov)
                    fourth = .5*(fourth+fourth.T)
                    fnp.fill_diagonal(fourth,post4)
                    fourth = flops.as_symmetric(fourth,symmetry=(0,1))
                gain = f[1][1]
                sources = [(born,gain[:,None]*aa,gain[:,None]*bb)
                           for born,aa,bb in sources if layer-born<self.history]
                if sources:
                    aa = fnp.concatenate([aa for born,aa,bb in sources],axis=1)
                    bb = fnp.concatenate([bb for born,aa,bb in sources],axis=1)
                    old3,oldp = third_slices(aa,bb)
                    post3 = post3-old3
                    repeated = repeated-oldp
                aa = 3*repeated.T
                fnp.fill_diagonal(aa,post3)
                sources.append((layer,aa,None))  # None represents an exact identity factor.
            else:
                sources = []
            cov = flops.as_symmetric(cov,symmetry=(0,1))
        return fnp.stack(rows)
