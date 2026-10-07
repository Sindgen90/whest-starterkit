import numpy as np
import flopscope.numpy as fnp
from screen import load

def call(name,m,v,t,k):
    mod=load(name);x=[fnp.asarray(np.atleast_1d(z),dtype=fnp.float32) for z in (m,v,t,k)];m,v,t,k=x
    raw,f,s,_,ph=mod.responses(m,v)
    lam=mod.coherent_factor(m,v,raw,f,s,ph,t,k) if name=='coherent' else fnp.asarray(1.,dtype=fnp.float32)
    out=mod.centered_update(m,v,raw,f,s,ph,t*lam,k*lam,coherent=name=='coherent')
    return [np.asarray(z,dtype=float) for z in out],float(np.asarray(lam))

def test_positive_identity_translation():
    for name in ('centered','coherent'):
        for shift in (10.,20.,1000.):
            out,lam=call(name,shift,.75,.75,-.375)
            np.testing.assert_allclose(out[0],shift,atol=1e-5)
            np.testing.assert_allclose(out[1],.75,atol=2e-6)
            np.testing.assert_allclose(out[2],lam*.75,rtol=3e-6)
            np.testing.assert_allclose(out[3],lam*(-.375),atol=3e-6)

def test_centered_gaussian_vs_independent_quadrature():
    # Hermite quadrature is inaccurate at the kink; split the Gaussian integral at 0.
    nodes,weights=np.polynomial.legendre.leggauss(384)
    for m in (-6.,-2.,0.,2.,6.,20.):
        x=(nodes+1)*6;ww=weights*6;h=np.maximum(m+x,0);hm=np.maximum(m-x,0)
        density=np.exp(-x*x/2)/np.sqrt(2*np.pi)
        mean=np.sum(ww*density*(h+hm));var=np.sum(ww*density*((h-mean)**2+(hm-mean)**2))
        third=np.sum(ww*density*((h-mean)**3+(hm-mean)**3));fourth=np.sum(ww*density*((h-mean)**4+(hm-mean)**4))-3*var*var
        # Integrate kink with a second, exactly split rule for tight checks.
        lo=max(-12.,-m);hi=12.;z=(nodes+1)*(hi-lo)/2+lo;wt=weights*(hi-lo)/2;dens=np.exp(-z*z/2)/np.sqrt(2*np.pi)
        raw=[np.sum(wt*dens*(m+z)**k) for k in (1,2,3,4)];mu,r2,r3,r4=raw
        exact=[mu,r2-mu*mu,r3-3*mu*r2+2*mu**3,r4-4*mu*r3-3*r2*r2+12*mu*mu*r2-6*mu**4]
        out,_=call('centered',m,1.,0.,0.)
        np.testing.assert_allclose(np.array(out).ravel(),exact,atol=1e-5,rtol=2e-4)

def test_centered_matches_unclipped_raw_algebra():
    mod=load('centered');rng=np.random.default_rng(22071);m=rng.uniform(-2,3,128);v=rng.uniform(.3,2,128);t=rng.normal(0,.1,128)*v**1.5;k=rng.normal(0,.1,128)*v*v
    arr=[fnp.asarray(x,dtype=fnp.float32) for x in (m,v,t,k)];raw,f,s,_,ph=mod.responses(*arr[:2]);d3,d4=arr[2:];delta=np.asarray(fnp.stack([f[q][3] for q in range(1,5)])*(d3/6)[None,:]+fnp.stack([f[q][4] for q in range(1,5)])*(d4/24)[None,:],dtype=float)
    bounds=.25*np.array([np.sqrt(v),v,v**1.5,v*v]);delta[:2]=np.clip(delta[:2],-bounds[:2],bounds[:2]);rr=np.asarray(fnp.stack(raw[1:]),dtype=float)+delta;mu,r2,r3,r4=rr
    expected=np.array([mu,r2-mu*mu,r3-3*mu*r2+2*mu**3,r4-4*mu*r3-3*r2*r2+12*mu*mu*r2-6*mu**4])
    got=np.array([np.asarray(x) for x in mod.centered_update(arr[0],arr[1],raw,f,s,ph,d3,d4)])
    np.testing.assert_allclose(got,expected,atol=2e-4,rtol=5e-3)
