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
    s,t,b,_=wb.shape
    v=(fnp.swapaxes(wb,2,3)@core.reshape((s,1,b,b*b*b))).reshape((s,t,b,b,b,b))
    v=fnp.transpose(v,(0,1,2,4,5,3)).reshape((s,t,b*b*b,b))@wb
    v=fnp.transpose(v.reshape((s,t,b,b,b,b)),(0,1,2,5,3,4))
    v=fnp.transpose(v,(0,1,2,3,5,4)).reshape((s,t,b*b*b,b))@wb
    v=fnp.transpose(v.reshape((s,t,b,b,b,b)),(0,1,2,3,5,4))
    v=(v.reshape((s,t,b*b*b,b))@wb).reshape((s,t,b,b,b,b))
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
