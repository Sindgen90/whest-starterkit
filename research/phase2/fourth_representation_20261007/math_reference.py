"""Independent NumPy finite-distribution/tensor oracle; offline only."""
import itertools
import numpy as np

def moments(x,prob=None):
    n,p=x.shape
    prob=np.full(n,1/n) if prob is None else np.asarray(prob)
    mu=prob@x;y=x-mu
    c=(y.T*prob)@y
    pairs=(y[:,:,None]*y[:,None,:]).reshape((n,p*p))
    t=((pairs.T*prob)@y).reshape((p,p,p))
    m4=((pairs.T*prob)@pairs).reshape((p,p,p,p))
    k=m4-c[:,:,None,None]*c[None,None,:,:]-c[:,None,:,None]*c[None,:,None,:]-c[:,None,None,:]*c[None,:,:,None]
    return mu,c,t,k

def kstatistics(x):
    n,p=x.shape;mu,c,t,k=moments(x)
    pair=c[:,:,None,None]*c[None,None,:,:]+c[:,None,:,None]*c[None,:,None,:]+c[:,None,None,:]*c[None,:,:,None]
    m4=k+pair
    ku=n*n*((n+1)*m4-(n-1)*pair)/((n-1)*(n-2)*(n-3))
    return mu,c*n/(n-1),t*n*n/((n-1)*(n-2)),ku

def pattern_masks(n):
    i,j,k,l=np.ogrid[:n,:n,:n,:n]
    qm=((i==j)&(k==l))|((i==k)&(j==l))|((i==l)&(j==k))
    rm=((i==j)&(j==k))|((i==j)&(j==l))|((i==k)&(k==l))|((j==k)&(k==l))
    return qm,rm

def projection(k,family,parts=None):
    n=k.shape[0];qm,rm=pattern_masks(n)
    base=np.where(qm|(rm if family in ('QR','F') else False),k,0) if family in ('Q','QR','E','F') else np.zeros_like(k)
    if parts is None:return base
    result=base.copy()
    for block in parts:
        ids=np.ix_(block,block,block,block);result[ids]+=k[ids]-base[ids]
    return result

def planes(k,w):
    n,m=w.shape;p=(w[:,None,:]*w[None,:,:]).reshape((n*n,m))
    middle=k.reshape((n*n,n*n))@p
    q=p.T@middle
    v=np.einsum('ija,ja->ia',middle.reshape((n,n,m)),w)
    r=v.T@w
    return np.diag(q),q,r

def full_transport(k,w):
    v=np.einsum('ijkl,ia->ajkl',k,w,optimize=True)
    v=np.einsum('ajkl,jb->abkl',v,w,optimize=True)
    v=np.einsum('abkl,kc->abcl',v,w,optimize=True)
    return np.einsum('abcl,ld->abcd',v,w,optimize=True)

def slices(k):
    ids=np.arange(len(k));return k[ids[:,None],ids[:,None],ids[None,:],ids[None,:]],k[ids[:,None],ids[:,None],ids[:,None],ids[None,:]]

def quadratic_matrix(k,c):
    ids=np.triu_indices(len(c));i,j=ids
    return k[i[:,None],j[:,None],i[None,:],j[None,:]]+c[i[:,None],i[None,:]]*c[j[:,None],j[None,:]]+c[i[:,None],j[None,:]]*c[j[:,None],i[None,:]]

def sym_error(k):return float(max(np.max(np.abs(k-k.transpose(p))) for p in itertools.permutations(range(4))))
