"""Independent mathematical identities and full-contract checks."""
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.generation import sample_mlp
from experiment import load,measure
from diagnose_allocation import tensor

def test_exact_dictionary_gaussian_integrals():
    mod,_=load('surrogate')
    mlp=sample_mlp(24,3,fnp.random.default_rng(913),seed=913)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        d,e=mod.dictionary(mlp);da=np.asarray(d);ea=np.asarray(e)
    assert ea.shape==(33,)
    np.testing.assert_allclose(ea[17:],np.linalg.norm(da,axis=0)/np.sqrt(2*np.pi),rtol=2e-7)
    np.testing.assert_array_equal(ea[1:17],0)

def test_surrogate_contract_and_immutability():
    mod,_=load('surrogate');mlp=sample_mlp(16,4,fnp.random.default_rng(914),seed=914)
    weights=np.asarray(mlp.weights).copy()
    pred,_=measure(mod.Estimator(),mlp)
    assert pred.shape==(4,16) and np.isfinite(pred).all()
    np.testing.assert_allclose(pred[0],np.linalg.norm(weights[0],axis=0)/np.sqrt(2*np.pi),rtol=2e-7)
    np.testing.assert_array_equal(np.asarray(mlp.weights),weights)

def test_distinct_score_matches_literal_tensor_contraction():
    mod,_=load('allocation');rng=np.random.default_rng(915)
    a=rng.normal(size=(5,3));b=rng.normal(size=(5,3));q=rng.normal(size=(5,4));c=rng.normal(size=4)
    exact=[]
    for r in range(3):
        d=tensor(a[:,r],b[:,r])
        values=np.einsum('ijk,ip,jp,kp->p',d,q,q,q)*c
        exact.append(np.linalg.norm(values))
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        actual=mod.distinct_scores(fnp.asarray(a),fnp.asarray(b),fnp.asarray(q),fnp.asarray(c))
        np.testing.assert_allclose(np.asarray(actual),exact,rtol=2e-13,atol=1e-12)

def test_individual_repair_keeps_repeated_tensor_entries():
    rng=np.random.default_rng(916);a=rng.normal(size=5);b=rng.normal(size=5)
    d=tensor(a,b)
    for i in range(5):
        np.testing.assert_array_equal(d[i,i,:],0)
        np.testing.assert_array_equal(d[i,:,i],0)
        np.testing.assert_array_equal(d[:,i,i],0)

def test_allocation_contract_and_immutability():
    mod,_=load('allocation');mlp=sample_mlp(16,6,fnp.random.default_rng(917),seed=917)
    weights=np.asarray(mlp.weights).copy();pred,_=measure(mod.Estimator(),mlp)
    assert pred.shape==(6,16) and np.isfinite(pred).all()
    np.testing.assert_array_equal(np.asarray(mlp.weights),weights)

def test_conditional_gaussian_decomposition_and_quadrature():
    rng=np.random.default_rng(918);u=rng.normal(size=7);u/=np.linalg.norm(u)
    c=np.eye(7)-np.outer(u,u)
    np.testing.assert_allclose(c@u,0,atol=4e-16)
    assert np.linalg.eigvalsh(c).min()>-1e-14
    z=np.array([0,np.sqrt(3),-np.sqrt(3)]);w=np.array([2/3,1/6,1/6])
    np.testing.assert_allclose((w*z).sum(),0,atol=1e-16)
    np.testing.assert_allclose((w*z*z).sum(),1,rtol=2e-16)
    np.testing.assert_allclose(c+np.outer(u,u)*(w*z*z).sum(),np.eye(7),atol=3e-16)

def test_conditional_exact_deterministic_neuron_over_three_identity_layers():
    from whestbench import MLP
    mod,_=load('conditional');n=4
    weights=[fnp.eye(n,dtype=fnp.float32) for _ in range(3)]
    mlp=MLP(n,3,weights,seed=919)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        u=fnp.asarray([1.,0.,0.,0.],dtype=fnp.float32)
        cov=fnp.eye(n,dtype=fnp.float32)-fnp.outer(u,u)
        for node in (0.,np.sqrt(3),-np.sqrt(3)):
            pred=np.asarray(mod.JointClosure().propagate(mlp,2**41,u*node,cov))
            np.testing.assert_allclose(pred[:,0],max(node,0),rtol=2e-7,atol=1e-7)
            np.testing.assert_allclose(pred[0,1:],1/np.sqrt(2*np.pi),rtol=3e-6,atol=1e-7)
            assert np.isfinite(pred).all()

def test_conditional_direction_and_contract():
    mod,_=load('conditional');mlp=sample_mlp(16,5,fnp.random.default_rng(920),seed=920)
    weights=np.asarray(mlp.weights).copy()
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        u=np.asarray(mod.important_direction(mlp,fnp.random.default_rng(921)))
    np.testing.assert_allclose(np.sum(u*u),1,rtol=3e-7)
    pred,_=measure(mod.Estimator(),mlp)
    assert pred.shape==(5,16) and np.isfinite(pred).all()
    np.testing.assert_array_equal(np.asarray(mlp.weights),weights)

def test_zero_gradient_direction_fallback_is_a_unit_vector():
    from whestbench import MLP
    mod,_=load('conditional');mlp=MLP(4,2,[fnp.zeros((4,4),dtype=fnp.float32)]*2,seed=922)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        u=np.asarray(mod.important_direction(mlp,fnp.random.default_rng(923)))
    np.testing.assert_allclose(u@u,1,rtol=3e-7)
