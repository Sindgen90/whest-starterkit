"""Metered marginal moments about c=max(m,0), with exact recenter identities."""
def centered_parts(m,v,moments,f,sigma,phi,d3,d4):
    c=fnp.maximum(m,0);c2=c*c;c3=c2*c;c4=c2*c2
    sf=flops.stats.norm.cdf(-m/sigma).astype(fnp.float32)
    sp=sigma*phi;positive=m>0;cdf=moments[0]
    # On z>0, h-m=z-m; on z<=0, h-m=-m. Avoid m^4 cancellation.
    g=fnp.stack((fnp.where(positive,sp-m*sf,moments[1]),
        fnp.where(positive,v*cdf-m*sp+c2*sf,moments[2]),
        fnp.where(positive,(c2+2*v)*sp-c3*sf,moments[3]),
        fnp.where(positive,3*v*v*cdf-(c3+3*m*v)*sp+c4*sf,moments[4])))
    d1=f[1][3]*(d3/6)+f[1][4]*(d4/24)
    d2=f[2][3]*(d3/6)+f[2][4]*(d4/24)
    h3=(6*cdf-6*c*f[1][2]+3*c2*f[1][3])*(d3/6)+(6*f[1][2]-6*c*f[1][3]+3*c2*f[1][4])*(d4/24)
    h4=(24*((m-c)*cdf+sp)+12*c2*f[1][2]-4*c3*f[1][3])*(d3/6)+(24*cdf-24*c*f[1][2]+12*c2*f[1][3]-4*c3*f[1][4])*(d4/24)
    h=fnp.stack((d1,d2-2*c*d1,h3,h4))
    return c,g,h,d1,d2

def central_recenter(c,r):
    r1,r2,r3,r4=(r[k] for k in range(4));r12=r1*r1
    variance=r2-r12
    third=r3-3*r1*r2+2*r12*r1
    fourth=r4-4*r1*r3-3*r2*r2+12*r12*r2-6*r12*r12
    return c+r1,fnp.maximum(variance,1e-12),third,fourth

def centered_update(m,v,moments,f,sigma,phi,d3,d4,coherent=False):
    c,g,h,d1,d2=centered_parts(m,v,moments,f,sigma,phi,d3,d4)
    if coherent:
        # All incoming cumulants have already received the same layer attenuation.
        return central_recenter(c,g+h)
    b1=.25*sigma;b2=.25*v
    clipped1=fnp.clip(d1,-b1,b1);clipped2=fnp.clip(d2,-b2,b2)
    e1=d1-clipped1;e2=d2-clipped2;c2=c*c
    # Preserve M1/M2 guards while leaving raw M3/M4 unguarded: exact shift algebra.
    shifted=fnp.stack((clipped1,clipped2-2*c*clipped1,
                      h[2]+3*c*e2-3*c2*e1,h[3]-6*c2*e2+4*c2*c*e1))
    return central_recenter(c,g+shifted)

def coherent_factor(m,v,moments,f,sigma,phi,d3,d4):
    _,_,h,_,_=centered_parts(m,v,moments,f,sigma,phi,d3,d4)
    bound=.25*fnp.stack((sigma,v,v*sigma,v*v))
    return fnp.minimum(1.,fnp.min(bound/fnp.maximum(fnp.abs(h),1e-30)))
