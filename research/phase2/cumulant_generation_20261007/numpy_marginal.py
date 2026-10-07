"""Offline float64 local update; no reference data are imported by estimators."""
import math
import numpy as np
def gaussian(m,v):
    s=np.sqrt(v);a=m/s;ph=np.exp(-.5*a*a)/np.sqrt(2*np.pi);cdf=np.fromiter((.5*math.erfc(-float(x)/np.sqrt(2)) for x in a),float,len(a));sf=np.fromiter((.5*math.erfc(float(x)/np.sqrt(2)) for x in a),float,len(a))
    mom=[cdf,m*cdf+s*ph];mom.append(m*mom[1]+v*cdf)
    for k in (3,4):mom.append(m*mom[-1]+(k-1)*v*mom[-2])
    f1=[None,cdf,ph/s,-a*ph/v,(a*a-1)*ph/(v*s)];f2=[None,2*mom[1],2*cdf,2*f1[2],2*f1[3]];f3=[None,None,None,6*cdf,6*f1[2]];f4=[None,None,None,24*mom[1],24*cdf]
    return np.array(mom),[None,f1,f2,f3,f4],s,ph,sf
def recenter(r,c=0):
    x,r2,r3,r4=r;v=r2-x*x;t=r3-3*x*r2+2*x**3;k=r4-4*x*r3-3*r2*r2+12*x*x*r2-6*x**4;return np.array([x+c,v,t,k])
def update(name,m,v,t,k):
    raw,f,s,ph,sf=gaussian(m,v);delta=np.array([f[q][3]*t/6+f[q][4]*k/24 for q in (1,2,3,4)]);bounds=.25*np.array([s,v,v*s,v*v]);mask=np.abs(delta)>bounds
    if name=='baseline_snapshot':return recenter(raw[1:]+np.clip(delta,-bounds,bounds)),mask,1.
    if name=='unclip34':
        d=delta.copy();d[:2]=np.clip(d[:2],-bounds[:2],bounds[:2]);return recenter(raw[1:]+d),mask,1.
    c=np.maximum(m,0);c2=c*c;c3=c2*c;c4=c2*c2;sp=s*ph;pos=m>0;cdf=raw[0]
    g=np.array([np.where(pos,sp-m*sf,raw[1]),np.where(pos,v*cdf-m*sp+c2*sf,raw[2]),np.where(pos,(c2+2*v)*sp-c3*sf,raw[3]),np.where(pos,3*v*v*cdf-(c3+3*m*v)*sp+c4*sf,raw[4])])
    h3=(6*cdf-6*c*f[1][2]+3*c2*f[1][3])*t/6+(6*f[1][2]-6*c*f[1][3]+3*c2*f[1][4])*k/24
    h4=(24*((m-c)*cdf+sp)+12*c2*f[1][2]-4*c3*f[1][3])*t/6+(24*cdf-24*c*f[1][2]+12*c2*f[1][3]-4*c3*f[1][4])*k/24
    h=np.array([delta[0],delta[1]-2*c*delta[0],h3,h4])
    if name=='coherent':
        lam=min(1.,float(np.min(bounds/np.maximum(np.abs(h),1e-30))));return recenter(g+h*lam,c),mask,lam
    cl=np.clip(delta[:2],-bounds[:2],bounds[:2]);e1,e2=delta[:2]-cl
    h=np.array([cl[0],cl[1]-2*c*cl[0],h[2]+3*c*e2-3*c2*e1,h[3]-6*c2*e2+4*c3*e1]);return recenter(g+h,c),mask,1.

def cumulants(d,pre=False):
    if pre:mu,m2,m3,m4=[d['pre_'+x] for x in ('mean','m2','m3','m4')]
    else:mu,m2,m3,m4=[d[x] for x in ('mean','post_m2','post_m3','post_m4')]
    return np.array([mu,m2-mu*mu,m3-3*mu*m2+2*mu**3,m4-4*mu*m3-3*m2*m2+12*mu*mu*m2-6*mu**4])
