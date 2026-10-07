"""Frozen joint moment candidate: twelve rich layers, four quadrature nodes.

Finite-memory third-cumulant factors and projected paired fourth cumulants.
Gaussian Cov(h,h), Cov(h^2,h), Cov(h^2,h^2) use Price-integral anchors.
Non-Gaussian response terms build on our local Puffi-inspired reconstruction;
this is not the published A716 algorithm. Fixed output-only readout fitted on public networks 0 and 1; never fed forward.
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
         4:((.06943184420297371,.17392742256872692),(.33000947820757187,.32607257743127305),
            (.6699905217924281,.32607257743127305),(.9305681557970262,.17392742256872692))}


def quadrature_tables(order):
    one = fnp.ones((1,1),dtype=fnp.float32)
    return tuple(fnp.stack([one*value for value in values]) for values in
                 ([x for x,w in QUADS[order]],[w*(1-x) for x,w in QUADS[order]],
                  [w*(1-x)**2 for x,w in QUADS[order]]))


def gaussian_pairs(cov,moments,sigma,a,phi,need_higher=True,order=8,tables=None):
    """Gaussian covariance identities evaluated with eight-point quadrature."""
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


def nong_terms(cov,pair3,pair4,d3,f,need_higher=True):
    """Batched response contractions; Gaussian terms are supplied separately."""
    pc = pair3*cov
    dc = (d3[:,None]*cov)/6
    kernels = fnp.stack((pair4*.25,pair3*.5,pair3.T*.5,pc*.5,pc.T*.5,
                        (pair3*pair3.T)*.25,dc,dc.T))
    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))
    left1 = fnp.stack([f[1][a] for a,b in orders])
    right1 = fnp.stack([f[1][b] for a,b in orders])
    if not need_higher:
        return fnp.einsum('ki,kij,kj->ij',left1,kernels,right1),None,None
    left2 = fnp.stack([f[2][a] for a,b in orders])
    right2 = fnp.stack([f[2][b] for a,b in orders])
    return tuple(fnp.einsum('ki,kij,kj->ij',left,kernels,right) for left,right in
                 ((left1,right1),(left2,right1),(left2,right2)))


class Estimator(BaseEstimator):
    history = 1
    use_fourth = True
    collect_diagnostics = False
    rich_layers = 12
    quadrature_order = 4
    batch_quadrature = True

    def predict(self,mlp,budget):
        n = mlp.width
        mu = fnp.zeros(n,dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n,dtype=fnp.float32),symmetry=(0,1))
        fourth = fnp.zeros((n,n),dtype=fnp.float32)
        sources = []
        rows = []
        designs = []
        self.trace = []
        zero = fnp.zeros(n,dtype=fnp.float32)
        tables = quadrature_tables(self.quadrature_order) if self.batch_quadrature else None
        for layer,w in enumerate(mlp.weights):
            m = mu@w
            pre = fnp.einsum('ij,ia,jb->ab',cov,w,w)
            v = fnp.maximum(fnp.diag(pre),1e-12)
            rich = layer<self.rich_layers
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
            mass = (fnp.sum(fnp.abs(pre),axis=1)-v)/n
            designs.append(fnp.stack((mu,v,a,mass,fnp.ones(n,dtype=fnp.float32)),axis=-1))
            if self.collect_diagnostics:
                mass = (fnp.sum(fnp.abs(pre),axis=1)-v)/n
                self.trace.append((m,v,d3,d4,mu,mass))
            if layer==len(mlp.weights)-1:
                break
            need_state = rich and layer+1<self.rich_layers
            g11,g21,g22 = gaussian_pairs(pre,moments,sigma,a,phi,need_higher=need_state,order=self.quadrature_order,tables=tables)
            if rich and layer>0:
                e11,e21,e22 = nong_terms(pre,pair3,pair4,d3,f,need_higher=need_state)
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
        coefficients = fnp.asarray(CALIBRATION,dtype=fnp.float32)
        return fnp.stack(rows)+fnp.einsum('lnk,lk->ln',fnp.stack(designs),coefficients)


CALIBRATION = [[7.3602037810209525e-06, 3.678222429102833e-06, 0.0, -0.00027407965132945726, 1.5685513874654e-06], [1.2945140491238339e-05, -2.4245506059259594e-05, -1.1531093344828573e-05, -0.0007140937657857808, 5.3085096961541414e-05], [-4.762514450137491e-06, -0.00011829082928422299, 3.8270470784395955e-06, -0.0012833277241770125, 0.0001721210291196009], [-7.775553939694114e-09, -0.00011332051477263186, -5.6838792369761e-06, -0.006461228940678597, 0.0003393880624357424], [-7.794764225688947e-06, -0.00017808820615582033, 8.11206391762476e-07, -0.003701760702354532, 0.0002539638758083466], [-1.0413302381653513e-05, -0.00011313566780634404, -5.377124816330599e-06, -0.010530543885003922, 0.00039107113038164905], [6.291264736545422e-05, -0.00019197374487435554, -1.558224759923538e-05, -0.006947464821141548, 0.00024044592886504474], [9.580442929823518e-05, -0.0004112797785937358, -3.4237690014776935e-05, -0.0068510220927263075, 0.00025002987540111523], [5.59884046636923e-05, 1.6667088965733304e-05, -2.5279914330100584e-05, -0.0036348957421991172, 1.1797363230970332e-05], [-4.750166342593455e-05, -0.0019286085777718773, -2.141274398146614e-06, 0.01093472720288939, 0.0003277063595954163], [2.8382549049979222e-05, -0.0006103096866141294, -1.488483896958568e-05, 0.0007760577825275727, 0.00010622759384727544], [1.0894905541710021e-05, -0.0015156512998555911, -1.008062086898768e-05, 0.008608481143260982, 0.00016034651156309992], [0.0005189414732143867, -0.002052371344530559, -0.0001360930967795376, -0.0012803196352399343, -0.0001389254657905983], [0.0002107840215911768, -0.0026724931762103997, -8.128289395061328e-05, 0.029687234054862, -0.0004980910164122719], [0.0001186362963807512, -0.002451096074250142, -6.737580519741674e-05, 0.015552732434991834, -0.00023220810524386576], [-0.00015234924940257265, -0.005004503011545734, 1.5521762018794535e-06, 0.05217432348837844, -0.0002832327387986859]]
