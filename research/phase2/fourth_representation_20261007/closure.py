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
    partition_rule='contiguous'
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
            if layer==len(mlp.weights)-1 and not self.collect_diagnostics:
                v=fnp.maximum(fnp.einsum('ij,ia,ja->a',cov,w,w),1e-12)
                d3=d4=zero
                if layer:
                    _,aa,bb=sources[0]
                    aa=w.T@aa;bb=w.T if bb is None else w.T@bb
                    d3=fnp.sum((aa*bb)*bb,axis=1)
                    if globalq:
                        square=w*w
                        diagonal=fnp.einsum('ij,ia,ja->a',fourth,square,square)
                        d4=3*diagonal-2*((square*square).T@fnp.diag(fourth))
                        if globalr:
                            roff=fnp.copy(rstate);fnp.fill_diagonal(roff,0)
                            d4=d4+4*fnp.sum((square*w)*(roff@w),axis=0)
                    if blocked:
                        residual=core
                        if globalq:residual=core-projected_core(local_plane(fourth,parts),local_plane(rstate,parts) if globalr else None,masks)
                        wb=w[parts,:]
                        pairmap=(wb[:,:,None,:]*wb[:,None,:,:]).reshape((nb,self.block_size*self.block_size,n))
                        middle=residual.reshape((nb,self.block_size*self.block_size,self.block_size*self.block_size))@pairmap
                        d4=d4+fnp.sum(pairmap*middle,axis=(0,1))
                rows.append(final_mean(m,v,d3,d4));break
            pre=w.T@(cov@w)
            pre=flops.as_symmetric((pre+pre.T)*.5,symmetry=(0,1))
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
