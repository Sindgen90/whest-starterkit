import itertools
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import kernels as m
from math_reference import moments,kstatistics,projection,planes,full_transport,slices,quadratic_matrix,sym_error

def test_raw_clipping_is_not_translation_consistent():
    from clipping_diagnostic import translation_example
    examples=translation_example()
    for e in examples:
        np.testing.assert_allclose(e['unclipped34_fourth'],e['true_input_and_output_fourth'],atol=1e-9)
        assert e['clipped_marginal_slack']<0
    assert abs(examples[0]['clipped_fourth']-examples[1]['clipped_fourth'])>1

def test_gaussian_connected_tree_leading_order():
    # Independent exact Gaussian polynomial quadrature, not the closure formula.
    nodes,weights=np.polynomial.hermite.hermgauss(7)
    grids=np.array(list(itertools.product(range(7),repeat=4)))
    x=np.sqrt(2)*nodes[grids];p=np.prod(weights[grids]/np.sqrt(np.pi),axis=1)
    base=np.array([[1.,.2,.1,-.12],[.2,1.,.18,.1],[.1,.18,1.,.15],[-.12,.1,.15,1.]])
    alpha=np.array([.3,-.2,.4,.25]);beta=np.array([.1,.3,-.2,.15]);eps=1e-3
    c=eps*base;z=x@np.linalg.cholesky(c).T
    h=z+alpha*z*z/2+beta*z*z*z/6
    truth=moments(h,p)[3]
    gain=1+beta*np.diag(c)/2
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        actual=np.asarray(m.nonlinear_core(fnp.zeros((1,4,4,4,4)),fnp.asarray(c[None]),fnp.zeros((1,4,4,4)),fnp.asarray(gain[None]),fnp.asarray(alpha[None]),fnp.asarray(beta[None])))[0]
    # Anchored repeated patterns have a different update; test four distinct only.
    np.testing.assert_allclose(actual[0,1,2,3],truth[0,1,2,3],rtol=.003,atol=1e-15)

def test_core_third_matches_literal_factor_sum():
    rng=np.random.default_rng(829);a=rng.normal(size=(6,3));b=rng.normal(size=(6,3));parts=np.array([[0,2,4],[1,3,5]])
    t=np.einsum('ir,jr,kr->ijk',a,b,b);t=(t+t.transpose(1,0,2)+t.transpose(2,1,0))/3
    with flops.BudgetContext(flop_budget=2**41,quiet=True):actual=np.asarray(m.third_core(fnp.asarray(a),fnp.asarray(b),fnp.asarray(parts)))
    np.testing.assert_allclose(actual,np.array([t[np.ix_(p,p,p)] for p in parts]),atol=1e-12)

def test_new_gaussian_cov31_against_independent_quadrature():
    from common import load
    mod=load('qr');c=np.array([[1.,.15],[.15,.8]],dtype=np.float32);mu=np.array([.4,.2],dtype=np.float32)
    nodes,weights=np.polynomial.hermite.hermgauss(100)
    grid=np.array(list(itertools.product(range(100),repeat=2)))
    x=np.sqrt(2)*nodes[grid]@np.linalg.cholesky(c).T+mu
    p=np.prod(weights[grid]/np.sqrt(np.pi),axis=1);y=np.maximum(x,0)
    expected=np.sum(p*y[:,0]**3*y[:,1])-np.sum(p*y[:,0]**3)*np.sum(p*y[:,1])
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        mm,ff,s,a,phi=mod.responses(fnp.asarray(mu),fnp.asarray(np.diag(c)))
        actual=np.asarray(mod.gaussian_extended(fnp.asarray(c),mm,s,a,phi,mod.quadrature_tables(2))[3])[0,1]
    np.testing.assert_allclose(actual,expected,rtol=.005,atol=.001)

def test_moment_conversion_and_translation():
    x=np.array([[-2.,.7],[1.,4.],[3.,-1.],[.2,2.5],[4.,1.]])
    p=np.array([.1,.2,.15,.25,.3]);mu,c,t,k=moments(x,p)
    for a in range(2):
        y=x[:,a]-mu[a]
        np.testing.assert_allclose(k[a,a,a,a],p@(y**4)-3*(p@(y*y))**2,atol=1e-12)
    shifted=moments(x+np.array([20.,-7.]),p)
    for a,b in zip((c,t,k),shifted[1:]):np.testing.assert_allclose(a,b,atol=5e-12)
    assert sym_error(k)<1e-12

def test_affine_full_cumulants_match_direct_finite_distribution():
    rng=np.random.default_rng(710);x=rng.normal(size=(11,4));p=rng.uniform(size=11);p/=p.sum();w=rng.normal(size=(4,3))
    mu,c,t,k=moments(x,p);m2,c2,t2,k2=moments(x@w,p)
    np.testing.assert_allclose(full_transport(k,w),k2,atol=3e-12)
    d,q,r=planes(k,w);q2,r2=slices(k2)
    np.testing.assert_allclose(q,q2,atol=3e-12);np.testing.assert_allclose(r,r2,atol=3e-12)

def test_nonsymmetric_r_multiplicities_and_all_equal_once():
    rng=np.random.default_rng(711);q=rng.normal(size=(4,4));q=(q+q.T)/2;r=rng.normal(size=(4,4));np.fill_diagonal(r,np.diag(q));w=rng.normal(size=(4,3))
    k=np.zeros((4,)*4)
    for i in range(4):
        k[i,i,i,i]=q[i,i]
        for j in range(i+1,4):
            for perm in set(itertools.permutations((i,i,j,j))):k[perm]=q[i,j]
        for j in range(4):
            if i!=j:
                for perm in set(itertools.permutations((i,i,i,j))):k[perm]=r[i,j]
    expected=planes(k,w)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        actual=m.global_planes(fnp.asarray(q),fnp.asarray(r),fnp.asarray(w),True)
        for a,e in zip(actual,expected):np.testing.assert_allclose(np.asarray(a),e,rtol=2e-13,atol=5e-12)
    paired=projection(k,'Q');d0,_,_=planes(paired,w)
    off=r.copy();np.fill_diagonal(off,0)
    np.testing.assert_allclose(expected[0]-d0,4*np.sum((w**3)*(off@w),axis=0),atol=5e-12)
    assert not np.allclose(r,r.T)

def test_block_overlap_subtraction_and_cross_block_covariance():
    rng=np.random.default_rng(712);_,c,_,k=moments(rng.normal(size=(30,4)))
    parts=np.array([[0,2],[1,3]])
    for family in ('D','E','F'):
        proj=projection(k,family,parts)
        for b in parts:np.testing.assert_allclose(proj[np.ix_(b,b,b,b)],k[np.ix_(b,b,b,b)],atol=1e-14)
    assert abs(c[0,1])>0 and projection(k,'D',parts)[0,0,1,1]==0

def test_metered_block_transports_against_literal_tensor():
    rng=np.random.default_rng(713);_,c,_,k=moments(rng.normal(size=(30,6)));w=rng.normal(size=(6,6));parts=np.array([[0,2,4],[1,3,5]])
    cores=np.stack([k[np.ix_(b,b,b,b)] for b in parts]);expected=projection(k,'D',parts)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        args=(fnp.asarray(cores),fnp.asarray(parts),fnp.asarray(w))
        actual=m.block_planes(*args);coreout=np.asarray(m.block_to_blocks(*args))
    for a,e in zip(actual,planes(expected,w)):np.testing.assert_allclose(np.asarray(a),e,rtol=5e-13,atol=1e-11)
    dense=full_transport(expected,w)
    for index,b in enumerate(parts):np.testing.assert_allclose(coreout[index],dense[np.ix_(b,b,b,b)],atol=1e-11)

def test_global_to_blocks_and_pattern_anchor():
    rng=np.random.default_rng(714);_,c,_,k=moments(rng.normal(size=(40,6)));q,r=slices(k);w=rng.normal(size=(6,6));parts=np.array([[0,2,4],[1,3,5]])
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        core=np.asarray(m.global_to_blocks(fnp.asarray(q),fnp.asarray(r),fnp.asarray(parts),fnp.asarray(w)))
        masks=m.pattern_masks(3)
        p=np.asarray(m.projected_core(fnp.asarray(q[parts[:,:,None],parts[:,None,:]]),fnp.asarray(r[parts[:,:,None],parts[:,None,:]]),masks))
    exact=full_transport(projection(k,'QR'),w)
    for ix,b in enumerate(parts):
        np.testing.assert_allclose(core[ix],exact[np.ix_(b,b,b,b)],atol=3e-11)
        np.testing.assert_allclose(p[ix],projection(k,'QR')[np.ix_(b,b,b,b)],atol=1e-14)

def test_symmetrization_and_quadratic_feature_repair():
    rng=np.random.default_rng(715);v=rng.normal(size=(4,)*4);literal=sum(v.transpose(p) for p in itertools.permutations(range(4)))/24
    c=np.eye(4);negative=-3*np.einsum('ij,kl->ijkl',c,c);negative=sum(negative.transpose(p) for p in itertools.permutations(range(4)))/24
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        np.testing.assert_allclose(np.asarray(m.symmetrize(fnp.asarray(v[None])))[0],literal,atol=1e-14)
        repaired,alpha=m.repair_core(fnp.asarray(negative[None]),fnp.asarray(c[None]));repaired=np.asarray(repaired)[0]
    assert np.linalg.eigvalsh(quadratic_matrix(repaired,c)).min()>-1e-8
    assert sym_error(repaired)<1e-12
    assert np.min([repaired[i,i,i,i]+2 for i in range(4)])>=-1e-8

def test_joint_kstat_affine_and_marginal_formula():
    rng=np.random.default_rng(716);x=rng.normal(size=(31,3));w=rng.normal(size=(3,2));mu,c,t,k=kstatistics(x)
    _,_,_,target=kstatistics(x@w);np.testing.assert_allclose(full_transport(k,w),target,atol=1e-11)
    y=x[:,0]-x[:,0].mean();n=len(y)
    expected=n*n*((n+1)*np.mean(y**4)-3*(n-1)*np.mean(y*y)**2)/((n-1)*(n-2)*(n-3))
    np.testing.assert_allclose(k[0,0,0,0],expected,atol=1e-13)
