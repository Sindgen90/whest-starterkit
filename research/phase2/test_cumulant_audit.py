"""Independent small-dimensional identities for the three focused ablations."""
import itertools
import numpy as np
import flopscope.numpy as fnp
from benchmark_cumulant_audit import load


def dense_third(a,b):
    return (np.einsum('ir,jr,kr->ijk',a,b,b)+np.einsum('ir,jr,kr->ijk',b,a,b)+np.einsum('ir,jr,kr->ijk',b,b,a))/3


def test_third_linear_transport_and_new_residual():
    mod,_ = load('baseline')
    rng = np.random.default_rng(61005)
    a,b,w = [rng.normal(size=(4,4)).astype('float32') for _ in range(3)]
    full = dense_third(a,b)
    transformed = np.einsum('ijk,ia,jb,kc->abc',full,w,w,w)
    transported = dense_third(w.T@a,w.T@b)
    np.testing.assert_allclose(transported,transformed,rtol=2e-6,atol=2e-6)
    keep = [1,3]
    old = dense_third(a[:,keep],b[:,keep])
    p = rng.normal(size=(4,4)).astype('float32')
    residual = p-np.einsum('iij->ij',old)
    source = 3*residual.T.copy()
    np.fill_diagonal(source,np.diag(residual))
    repaired = old+dense_third(source,np.eye(4))
    np.testing.assert_allclose(np.einsum('iij->ij',repaired),p,rtol=2e-6,atol=2e-6)
    assert np.any(np.abs(old[0,1,2])>0)
    assert dense_third(source,np.eye(4))[0,1,2]==0
    diag, repeated = mod.third_slices(fnp.asarray(a),fnp.asarray(b))
    np.testing.assert_allclose(repeated,np.einsum('iij->ij',full),rtol=2e-6,atol=2e-6)


def test_exact_recentering_and_gaussian_pairing_subtraction():
    rng = np.random.default_rng(61006)
    h = rng.normal(size=(31,4))**2+np.array([1.,-2.,3.,.5])
    mu = h.mean(0)
    y = h-mu
    c = y.T@y/len(h)
    c21 = ((h*h).T@h)/len(h)-np.outer((h*h).mean(0),mu)
    c22 = ((h*h).T@(h*h))/len(h)-np.outer((h*h).mean(0),(h*h).mean(0))
    p = c21-2*mu[:,None]*c
    q = c22-2*mu[None,:]*c21-2*mu[:,None]*c21.T+4*np.outer(mu,mu)*c-2*c*c
    np.testing.assert_allclose(p,(y*y).T@y/len(h),rtol=1e-12,atol=1e-12)
    true_q = (y*y).T@(y*y)/len(h)-np.outer(np.diag(c),np.diag(c))-2*c*c
    np.testing.assert_allclose(q,true_q,rtol=1e-12,atol=1e-12)


def test_cross_mode_matches_explicit_paired_tensor():
    mod,_ = load('fourth_cross1')
    rng = np.random.default_rng(61007)
    q = rng.normal(size=(4,4)).astype('float32')
    q = (q+q.T)/2
    w = rng.normal(size=(4,3)).astype('float32')
    off = q.copy()
    np.fill_diagonal(off,0)
    u = np.ones(4,dtype='float32')/2
    for _ in range(8):
        u = off@u
        u /= max(np.sqrt(np.sum(u*u)),1e-20)
    lam = np.sum(u*(off@u))
    approximation = lam*np.outer(u,u)
    np.fill_diagonal(approximation,0)
    tensor = np.zeros((4,4,4,4))  # Proof only, never constructed by an estimator.
    for i in range(4):
        for j in range(i):
            for indices in set(itertools.permutations((i,i,j,j))):
                tensor[indices] = approximation[i,j]
    actual = np.einsum('ijkl,ia,ja,kb,lb->ab',tensor,w,w,w,w)
    gram = (w*w).T@approximation@(w*w)
    got = mod.rank_one_cross(fnp.zeros((3,3),dtype=fnp.float32),fnp.asarray(q),fnp.asarray(w))
    np.testing.assert_allclose(got,actual-gram,rtol=3e-5,atol=2e-5)


def test_cross_mode_zero_is_finite_and_vanishes():
    mod,_ = load('fourth_cross1')
    q = fnp.zeros((4,4),dtype=fnp.float32)
    w = fnp.eye(4,dtype=fnp.float32)
    result = mod.rank_one_cross(q,q,w)
    np.testing.assert_array_equal(result,np.zeros((4,4)))


def test_marginal_fourth_response_coefficient_and_zero_ablation():
    mod,_ = load('fourth_response')
    base,_ = load('baseline')
    m = fnp.asarray([-.2,.7],dtype=fnp.float32)
    v = fnp.asarray([.8,1.1],dtype=fnp.float32)
    _,f,*_ = mod.responses(m,v)
    _,original,*_ = base.responses(m,v)
    c = fnp.asarray([[.8,.03],[.03,1.1]],dtype=fnp.float32)
    p = fnp.asarray([[.1,.02],[-.04,.1]],dtype=fnp.float32)
    q = p+p.T
    d3 = fnp.asarray([.1,-.2],dtype=fnp.float32)
    d4 = fnp.asarray([.3,-.4],dtype=fnp.float32)
    old = base.nong_terms(c,p,q,d3,original)
    zero = mod.nong_terms(c,p,q,d3,f,d4=0*d4)
    new = mod.nong_terms(c,p,q,d3,f,d4=d4)
    for k,(left,right) in enumerate(((1,1),(2,1),(2,2))):
        np.testing.assert_allclose(zero[k],old[k],rtol=2e-6,atol=1e-8)
        expected = np.asarray(c)*(np.outer(np.asarray(d4*f[left][5]),np.asarray(f[right][1]))+np.outer(np.asarray(f[left][1]),np.asarray(d4*f[right][5])))/24
        np.testing.assert_allclose(np.asarray(new[k])-np.asarray(old[k]),expected,rtol=3e-5,atol=1e-8)
    # The weak-covariance term is exact for f(x)=x^5, g(y)=y:
    # (k4/24)*E[d_x^4(x^5*y)] = (k4/24)*120*Cov(x,y).
    assert 120/24==5


def test_mixed_third_squares_have_the_same_order_as_retained_product():
    mod,_ = load('third_square')
    base,_ = load('baseline')
    _,f,*_ = base.responses(fnp.asarray([-.2,.7],dtype=fnp.float32),fnp.asarray([.8,1.1],dtype=fnp.float32))
    c = fnp.asarray([[.8,.03],[.03,1.1]],dtype=fnp.float32)
    p = fnp.asarray([[.1,.02],[-.04,.1]],dtype=fnp.float32)
    q = p+p.T
    d3 = fnp.asarray([.1,-.2],dtype=fnp.float32)
    old = base.nong_terms(c,p,q,d3,f)
    new = mod.nong_terms(c,p,q,d3,f)
    for k,(left,right) in enumerate(((1,1),(2,1),(2,2))):
        expected = (np.asarray(p)**2*np.outer(np.asarray(f[left][4]),np.asarray(f[right][2]))+np.asarray(p.T)**2*np.outer(np.asarray(f[left][2]),np.asarray(f[right][4])))/8
        np.testing.assert_allclose(np.asarray(new[k])-np.asarray(old[k]),expected,rtol=3e-5,atol=1e-8)


def test_candidates_repeat_without_mutation_and_match_full_terminal_state():
    import flopscope as flops
    from whestbench import MLP
    rng = np.random.default_rng(61008)
    raw = (rng.normal(size=(6,12,12))*np.sqrt(2/12)).astype('float32')
    for name in ('third_memory16','fourth_cross1','fourth_response','third_square'):
        mod,_ = load(name)
        weights = fnp.asarray(raw.copy())
        mlp = MLP(12,6,list(weights),seed=0)
        est = mod.Estimator()
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            first = np.asarray(est.predict(mlp,2**41)).copy()
            second = np.asarray(est.predict(mlp,2**41)).copy()
            est.collect_diagnostics = True
            full = np.asarray(est.predict(mlp,2**41)).copy()
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(np.asarray(weights),raw)
        np.testing.assert_allclose(full,first,rtol=5e-6,atol=3e-6)
